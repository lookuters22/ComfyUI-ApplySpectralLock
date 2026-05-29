# ComfyUI-ApplySpectralLock

**Apply Spectral Lock (Flux/USDU)** — a ComfyUI model patch that performs a **Latent Spectral Handoff** on **16-channel Flux** latents during sampling. It uses two **decoupled** controls:

- **Structure lock** — low frequencies are anchored to a reference latent (`latent_orig`). Active from step 0, so composition / color / lighting are preserved even at 0.8 denoise.
- **Detail gate (variance matching)** — instead of blanket dampening, the high frequencies are **renormalized to match the reference's per-channel contrast/energy**. This keeps real, crisp detail while stopping the Rectified Flow solver from overshooting into digital grain ("Turbo paradox"). The gate follows a **schedule-aware cosine (or linear) envelope**, engaging mostly over the final steps.

## Requirements

- ComfyUI (recent master)
- PyTorch
- **Flux-family models** with **16 latent channels** (SD 4-channel latents are passed through unchanged)

## Install

Clone into `ComfyUI/custom_nodes/`:

```text
ComfyUI/custom_nodes/ComfyUI-ApplySpectralLock/
```

Restart ComfyUI.

## Usage

1. Load your **MODEL** as usual.
2. Provide **`latent_orig`**: the full-resolution (or reference) latent corresponding to the uncorrupted base structure you want to lock to (image2latent / VAE encode of the guide image).
3. Run **ApplySpectralLock** → connect **`patched_model`** into your sampler / Ultimate SD Upscale chain **before** sampling.

### Parameters

| Input | Role |
|--------|------|
| `kernel_size` | Odd kernel size for separable Gaussian low-pass (default 15). |
| `blur_sigma` | Gaussian sigma for the 1D kernels (default 3.0). |
| `alpha_lock` | Structure lock: blend weight for anchoring low frequencies to `latent_orig` (0–1, default 0.85). Active from step 0. |
| `gamma_max` | Detail gate: maximum blend toward the soft-clipped high band over the final steps (0–1, default 0.70). Higher = stronger grain suppression. |
| `clip_k` | Soft-clip knee in units of the high-band sigma (default 2.0). Spikes beyond `k*sigma` are tamed; ordinary detail (`|H| << knee`) passes through. Lower = more aggressive grain removal; higher = gentler / more detail kept. |
| `decay_type` | `cosine` (default, slow start then firm asymptote) or `linear` progress scaling for the detail gate. |
| `edge_taper` | Spatial edge-gating width in latent cells (default 48, `0` disables). The spectral adjustment is cosine-tapered to zero at the tile border so USDU seam blending sees no discontinuity. Auto-clamped to half the tile size. |

### Ultimate SD Upscale (USDU)

For tiled runs, the patcher estimates the tile’s **(y, x)** offset in latent space via **normalized template matching** on **channel-mean maps**, then crops `latent_orig` once per tile and caches the blurred reference low-pass for the rest of that tile’s steps.

**Note:** Very large Gaussian radii relative to USDU overlap can still stress seams; keep `kernel_size` coherent with your tile overlap.

### Flux CFG = 1.0

The hook is registered with **`disable_cfg1_optimization=True`** so the post-CFG callback still runs when CFG is disabled (common for distilled / single-stream Flux runs).

## Implementation notes

- **Depthwise separable** Gaussian: two `F.conv2d` passes with `groups=16`, **`F.pad(..., mode='reflect')`** before each pass.
- **Detail-preserving grain suppression**: the high band is soft-clipped with a `tanh` knee at `clip_k * sigma`. Grain/"deep-fry" is sparse extreme spikes, so clipping tames them while ordinary structured detail (`|H| << knee`) passes through unchanged — it does NOT normalize energy toward the reference (which would blur detail down to a soft base).
- **Spatial edge-gating**: a cached cosine taper mask zeroes the spectral delta `(X_new - denoised)` at tile borders (`edge_taper`), so USDU seam blending never reconciles a discontinuity introduced here.
- **Decoupled envelopes**: structure lock (`alpha`) is applied every step; the variance gate (`gamma`) ramps with schedule progress.
- **Float32** for blur / variance / NCC internals; outputs are cast back to **`denoised.dtype`** and device.
- **Sigma progress**: schedule-agnostic `progress = 1 - sigma / max_sigma_seen` for the current tile/run.

## License

[MIT](LICENSE) (c) 2026 lookuters22.
