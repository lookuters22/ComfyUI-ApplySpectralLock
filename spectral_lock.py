"""
Latent Spectral Handoff for Flux.1 / Rectified Flow.

Depthwise separable Gaussian low-pass splits each denoised latent into low (L) and
high (H) frequency bands. Two decoupled controls are then applied:

  * Structure lock (alpha): low frequencies are anchored to a reference latent.
    Active from step 0 to preserve composition / color / lighting.
  * Detail gate (gamma, cosine envelope): instead of blanket dampening, the high
    frequencies are *variance matched* to the reference's per-channel energy, so
    generated texture keeps real detail while the solver is prevented from
    overshooting into digital grain. Engages mostly over the final steps. A joint
    (cross-channel) epsilon keeps per-channel scaling coherent to avoid color drift.

Latent-space normalized cross-correlation aligns USDU sub-tiles to the reference,
and a cosine spatial edge-taper forces the adjustment to zero at tile borders so
USDU seam blending stays clean.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

import comfy.model_patcher


class SeparableGaussianSmoothing(torch.nn.Module):
    """
    Depthwise separable 1D Gaussian smoothing for 16-channel Flux latents.
    Each channel is filtered in isolation (groups=C). Two 1D passes approximate
    a 2D Gaussian with lower cost than a full 2D conv.
    """

    def __init__(
        self,
        channels: int = 16,
        kernel_size: int = 15,
        sigma: float = 3.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding.")
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.channels = int(channels)

        x_cord = torch.arange(self.kernel_size, dtype=torch.float32)
        x_grid = x_cord - float(self.padding)
        gaussian_1d = torch.exp(-(x_grid**2) / (2.0 * float(sigma) ** 2))
        gaussian_1d = gaussian_1d / torch.sum(gaussian_1d)

        kernel_x = gaussian_1d.view(1, 1, -1, 1).repeat(self.channels, 1, 1, 1)
        kernel_y = gaussian_1d.view(1, 1, 1, -1).repeat(self.channels, 1, 1, 1)

        self.register_buffer("kernel_x", kernel_x.to(dtype=dtype))
        self.register_buffer("kernel_y", kernel_y.to(dtype=dtype))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        device = x.device
        kx = self.kernel_x.to(device=device, dtype=torch.float32)
        ky = self.kernel_y.to(device=device, dtype=torch.float32)
        x32 = x.to(dtype=torch.float32)

        x_padded = F.pad(
            x32,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        out = F.conv2d(x_padded, kx, groups=self.channels)
        out = F.conv2d(out, ky, groups=self.channels)
        return out.to(dtype=original_dtype)


def _find_tile_offset_ncc(full: torch.Tensor, tile: torch.Tensor) -> Tuple[int, int]:
    """
    Normalized cross-correlation in latent space: channel-mean maps, ZNCC-style
    template as conv weight, argmax of correlation map -> (y, x) top-left of tile in full.
    """
    if full.ndim != 4 or tile.ndim != 4:
        return 0, 0
    b_f, _, hf, wf = full.shape
    b_t, _, ht, wt = tile.shape
    if b_f < 1 or b_t < 1 or ht > hf or wt > wf or ht < 1 or wt < 1:
        return 0, 0

    full_32 = full[0:1].to(dtype=torch.float32).mean(dim=1, keepdim=True)
    tile_32 = tile[0:1].to(dtype=torch.float32).mean(dim=1, keepdim=True)

    tile_32 = tile_32 - tile_32.mean()
    norm = torch.sqrt(torch.sum(tile_32**2))
    if float(norm.item()) <= 1e-6:
        return 0, 0
    tile_32 = tile_32 / norm

    weight = tile_32  # [1,1,ht,wt]
    corr = F.conv2d(full_32, weight)  # [1,1, hf-ht+1, wf-wt+1]
    flat = corr.view(-1)
    if flat.numel() == 0:
        return 0, 0
    max_idx = int(torch.argmax(flat).item())
    ow = int(corr.size(3))
    y = max_idx // ow
    x = max_idx % ow
    y = max(0, min(y, hf - ht))
    x = max(0, min(x, wf - wt))
    return y, x


def _build_edge_taper_mask(
    h: int,
    w: int,
    taper: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    2D separable cosine edge-taper mask: 1.0 in the interior, smoothly cosine-faded
    to exactly 0.0 at the absolute tile border over the outer `taper` rows/cols.

    Forces any spectral modification to vanish at tile edges so the USDU blend mask
    never has to reconcile a discontinuity introduced by this node. Returns [1,1,H,W].
    """
    taper = int(max(0, min(taper, h // 2, w // 2)))
    if taper <= 0:
        return torch.ones((1, 1, h, w), device=device, dtype=dtype)

    def _ramp(n: int) -> torch.Tensor:
        idx = torch.arange(n, device=device, dtype=torch.float32)
        edge_dist = torch.minimum(idx, (n - 1) - idx)
        r = torch.ones(n, device=device, dtype=torch.float32)
        m = edge_dist < taper
        # 0 at the edge (edge_dist=0) -> 1 at edge_dist=taper, smooth cosine in between.
        r[m] = 0.5 * (1.0 - torch.cos(math.pi * edge_dist[m] / float(taper)))
        return r

    ry = _ramp(h)
    rx = _ramp(w)
    mask = torch.outer(ry, rx)  # [H, W]
    return mask.view(1, 1, h, w).to(dtype=dtype)


class SpectralLockPatcher:
    """Model patcher node: injects post-CFG hook for latent spectral handoff."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "latent_orig": ("LATENT",),
                "kernel_size": ("INT", {"default": 15, "min": 3, "max": 63, "step": 2}),
                "blur_sigma": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "alpha_lock": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma_max": ("FLOAT", {"default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01}),
                "decay_type": (["cosine", "linear"],),
                "edge_taper": ("INT", {"default": 48, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/spectral"

    def patch(
        self,
        model: Any,
        latent_orig: Dict[str, Any],
        kernel_size: int,
        blur_sigma: float,
        alpha_lock: float,
        gamma_max: float,
        decay_type: str,
        edge_taper: int = 48,
    ) -> Tuple[Any]:
        m = model.clone()
        X_orig_full = latent_orig["samples"]

        state: Dict[str, Any] = {
            "max_sigma": None,
            "gaussian_filter": None,
            "cached_L_orig": None,
            "cached_H_orig_std": None,
            "cached_edge_mask": None,
            "cached_tile_shape": None,
            "last_sigma": None,
        }

        def post_cfg_function(args: Dict[str, Any]) -> torch.Tensor:
            denoised = args["denoised"]
            if not isinstance(denoised, torch.Tensor):
                return denoised
            if denoised.ndim != 4 or denoised.shape[1] != 16:
                return denoised
            if not isinstance(X_orig_full, torch.Tensor) or X_orig_full.ndim != 4:
                return denoised
            if int(X_orig_full.shape[1]) != 16:
                return denoised

            sigma_tensor = args["sigma"]
            if isinstance(sigma_tensor, torch.Tensor):
                sigma_val = float(sigma_tensor.detach().float().max().item())
            else:
                sigma_val = float(sigma_tensor)

            prev = state["last_sigma"]
            is_new_run = prev is None or sigma_val > prev + 1e-4
            if is_new_run:
                state["max_sigma"] = None
                state["cached_L_orig"] = None
                state["cached_H_orig_std"] = None
                state["cached_edge_mask"] = None
                state["cached_tile_shape"] = None
            state["last_sigma"] = sigma_val

            if state["max_sigma"] is None or sigma_val > float(state["max_sigma"]):
                state["max_sigma"] = sigma_val

            progress = 0.0
            max_sigma = state["max_sigma"]
            if max_sigma is not None and float(max_sigma) > 0:
                progress = 1.0 - (sigma_val / float(max_sigma))
                progress = max(0.0, min(1.0, progress))

            if decay_type == "cosine":
                gamma_t = float(gamma_max) * 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                gamma_t = float(gamma_max) * progress

            if gamma_t < 0.01 and float(alpha_lock) < 0.01:
                return denoised

            device = denoised.device
            dtype = denoised.dtype

            if state["gaussian_filter"] is None:
                state["gaussian_filter"] = SeparableGaussianSmoothing(
                    channels=16,
                    kernel_size=int(kernel_size),
                    sigma=float(blur_sigma),
                    dtype=torch.float32,
                ).to(device=device)

            gaussian: SeparableGaussianSmoothing = state["gaussian_filter"]
            if gaussian.kernel_x.device != device:
                gaussian.to(device=device)

            _, _, h, w = denoised.shape
            need_match = state["cached_L_orig"] is None or state["cached_tile_shape"] != (h, w)

            if need_match:
                X_orig = X_orig_full.to(device=device, dtype=dtype)
                oh, ow = int(X_orig.shape[2]), int(X_orig.shape[3])

                if oh > h or ow > w:
                    input_noisy = args.get("input")
                    if isinstance(input_noisy, torch.Tensor) and input_noisy.shape == denoised.shape:
                        y_off, x_off = _find_tile_offset_ncc(X_orig, input_noisy)
                    else:
                        y_off, x_off = 0, 0
                    y_end = min(y_off + h, oh)
                    x_end = min(x_off + w, ow)
                    y_off = max(0, y_end - h)
                    x_off = max(0, x_end - w)
                    X_orig_cropped = X_orig[:, :, y_off : y_off + h, x_off : x_off + w]
                else:
                    X_orig_cropped = X_orig

                if X_orig_cropped.shape[2] != h or X_orig_cropped.shape[3] != w:
                    X_orig_cropped = F.interpolate(
                        X_orig_cropped.float(),
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    ).to(dtype=dtype)

                L_orig = gaussian(X_orig_cropped)
                # High-frequency residual of the reference tile and its per-channel
                # spatial energy (std). Computed once per tile in fp32 for stability.
                H_orig_32 = (X_orig_cropped.to(torch.float32) - L_orig.to(torch.float32))
                H_orig_std = H_orig_32.std(dim=(2, 3), keepdim=True, unbiased=False)
                state["cached_L_orig"] = L_orig.to(dtype=dtype)
                state["cached_H_orig_std"] = H_orig_std  # keep in fp32
                # Cosine edge-taper mask for this tile geometry (fp32, broadcast over C).
                state["cached_edge_mask"] = _build_edge_taper_mask(
                    h, w, int(edge_taper), device, torch.float32
                )
                state["cached_tile_shape"] = (h, w)

            L_t = gaussian(denoised).to(dtype=dtype)
            H_t = denoised - L_t
            L_orig_tile = state["cached_L_orig"]
            H_orig_std = state["cached_H_orig_std"]
            if L_orig_tile is None or H_orig_std is None:
                return denoised

            # --- Structure lock (decoupled): fully active from step 0 ---
            # Anchors low frequencies (composition/color/lighting) to the reference.
            alpha = float(alpha_lock)
            L_new = (L_orig_tile * alpha) + (L_t * (1.0 - alpha))

            # --- Detail gate (decoupled): variance matching governed by gamma_t ---
            # Instead of shrinking high-frequency energy, renormalize the generated
            # texture so its per-channel contrast/energy matches the reference latent.
            # This preserves real detail while preventing the Euler solver from
            # overshooting into digital grain. The cosine envelope (gamma_t) controls
            # how strongly this governance engages, ramping up over the final steps.
            H_t_32 = H_t.to(torch.float32)
            std_t = H_t_32.std(dim=(2, 3), keepdim=True, unbiased=False)

            # Covariance-preserving (joint) epsilon: tie the denominator floor to the
            # mean energy ACROSS the 16 channels, not to each channel in isolation. A
            # near-flat channel therefore inherits the joint floor instead of being
            # independently amplified, which keeps the inter-channel scale ratios stable
            # and prevents microscopic color/tonal drift in the Flux latent.
            eps_abs = 1e-4
            eps_rel = 0.10
            joint_energy = std_t.mean(dim=1, keepdim=True)  # [B,1,1,1] across channels
            eps_joint = eps_abs + eps_rel * joint_energy
            scale = H_orig_std / (std_t + eps_joint)
            # Clamp the gain so a near-flat generated tile can't be blown up into noise.
            scale = torch.clamp(scale, max=10.0)

            H_matched = H_t_32 * scale
            # Blend raw texture -> variance-matched texture as governance engages.
            H_new = H_t_32 * (1.0 - gamma_t) + H_matched * gamma_t
            H_new = H_new.to(dtype=dtype)

            X_new = L_new + H_new

            # --- Spatial edge-gating ---
            # Force the entire spectral adjustment to zero at the absolute tile border
            # via a cosine taper, so USDU tile blending sees no seam discontinuity.
            edge_mask = state.get("cached_edge_mask")
            if edge_mask is not None:
                if edge_mask.device != device:
                    edge_mask = edge_mask.to(device=device)
                    state["cached_edge_mask"] = edge_mask
                delta = (X_new.to(torch.float32) - denoised.to(torch.float32)) * edge_mask
                out = denoised.to(torch.float32) + delta
                return out.to(dtype=dtype, device=device)

            return X_new.to(dtype=dtype, device=device)

        model_options = dict(m.model_options)
        m.model_options = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options,
            post_cfg_function,
            disable_cfg1_optimization=True,
        )
        return (m,)
