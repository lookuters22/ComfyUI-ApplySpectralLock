"""Advanced Latent Spectral Handoff node for ComfyUI.

Optimized for Flux.1 Rectified Flow architectures and tiled refinement (USDU).
"""

from .spectral_lock import SpectralLockPatcher

NODE_CLASS_MAPPINGS = {
    "ApplySpectralLock": SpectralLockPatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplySpectralLock": "Apply Spectral Lock (Flux/USDU)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
