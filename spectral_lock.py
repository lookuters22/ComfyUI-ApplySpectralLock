"""
Latent Spectral Handoff for Flux.1 / Rectified Flow.

Depthwise separable Gaussian low-pass splits each denoised latent into low (L) and
high (H) frequency bands. Two decoupled controls are then applied:

  * Structure lock (alpha): low frequencies are anchored to a reference latent.
    Active from step 0 to preserve composition / color / lighting.
  * Detail-preserving grain suppression (gamma, cosine envelope): the high band is
    soft-clipped with a tanh knee at k*sigma. Sparse extreme spikes (grain / "deep-fry")
    are tamed while ordinary structured detail passes through unchanged, so detail is
    preserved rather than blurred. Engages over the final steps via the cosine gate.

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
                "clip_k": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 6.0, "step": 0.1}),
                "decay_type": (["cosine", "linear"],),
                "edge_taper": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
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
        edge_taper: int = 8,
        clip_k: float = 2.0,
    ) -> Tuple[Any]:
        m = model.clone()
        X_orig_full = latent_orig["samples"]

        state: Dict[str, Any] = {
            "max_sigma": None,
            "gaussian_filter": None,
            "cached_L_orig": None,
            "cached_X_orig_cropped": None,
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
                state["cached_X_orig_cropped"] = None
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
                    # FIX 1: match against the clean denoised output, not args["input"]
                    # (the noisy latent), which gives unstable/random crop coordinates.
                    y_off, x_off = _find_tile_offset_ncc(X_orig, denoised)
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
                # Cache the aligned reference tile and its low-pass; the high-frequency
                # std is recomputed per step in the robust variance-matching block.
                state["cached_L_orig"] = L_orig.to(dtype=dtype)
                state["cached_X_orig_cropped"] = X_orig_cropped.to(dtype=dtype)
                # Cosine edge-taper mask for this tile geometry (fp32, broadcast over C).
                state["cached_edge_mask"] = _build_edge_taper_mask(
                    h, w, int(edge_taper), device, torch.float32
                )
                state["cached_tile_shape"] = (h, w)

            L_t = gaussian(denoised).to(dtype=dtype)
            H_t = denoised - L_t
            L_orig_tile = state["cached_L_orig"]
            if L_orig_tile is None:
                return denoised

            # --- Structure lock (decoupled): fully active from step 0 ---
            # Anchors low frequencies (composition/color/lighting) to the reference.
            alpha = float(alpha_lock)
            L_new = (L_orig_tile * alpha) + (L_t * (1.0 - alpha))

            # --- Detail-preserving grain suppression (soft-clip on HF outliers) ---
            # KEY: do NOT normalize energy toward the (soft) reference -- that erases
            # the detail the refiner adds and blurs the result. Grain / "deep-fry" is
            # sparse EXTREME spikes in the high band, whereas genuine detail is moderate.
            # A tanh soft-knee at k*sigma tames the spikes while passing ordinary detail
            # through unchanged (tanh(x)~x for |x| << knee), so detail is preserved.
            H_t_32 = H_t.to(torch.float32)
            sigma_h = H_t_32.std(dim=[2, 3], keepdim=True)  # per-channel spatial spread
            knee = (float(clip_k) * sigma_h) + 1e-4
            H_soft = knee * torch.tanh(H_t_32 / knee)

            # Cosine gate: no change early (gamma_t~0), clip engages over the final steps.
            # gamma_t already folds in gamma_max as the maximum blend toward the clipped HF.
            H_new = (H_t_32 + gamma_t * (H_soft - H_t_32)).to(dtype)

            # --- Spatial edge-gating (FIX 3) ---
            # Force the entire spectral adjustment to zero at the absolute tile border
            # via a cosine taper (fp32), so USDU tile blending sees no seam discontinuity.
            combined = L_new + H_new
            edge_mask = state.get("cached_edge_mask")
            if edge_mask is not None:
                if edge_mask.device != device:
                    edge_mask = edge_mask.to(device=device)
                    state["cached_edge_mask"] = edge_mask
                mask = edge_mask.to(torch.float32)
                X_new = denoised.to(torch.float32) + (
                    combined.to(torch.float32) - denoised.to(torch.float32)
                ) * mask
                return X_new.to(dtype=dtype, device=device)

            return combined.to(dtype=dtype, device=device)

        model_options = dict(m.model_options)
        m.model_options = comfy.model_patcher.set_model_options_post_cfg_function(
            model_options,
            post_cfg_function,
            disable_cfg1_optimization=True,
        )
        return (m,)
