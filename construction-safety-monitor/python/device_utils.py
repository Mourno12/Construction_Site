"""
device_utils.py
Picks the best available torch device: CUDA (NVIDIA) > XPU (Intel Arc /
Data Center GPU, via PyTorch's native Intel GPU backend) > CPU. Centralized
here so training scripts (train_vit.py) and the live ViT inference path
(ppe_classifier.py) all pick the same device the same way, instead of each
hardcoding "cuda" and silently falling back to CPU on machines whose only
GPU is Intel's.

Note: the plain `pip install torch` wheel from PyPI is CPU-only. Getting
`torch.xpu.is_available()` to return True requires installing torch from
Intel/PyTorch's XPU-specific index instead:
    pip install torch --index-url https://download.pytorch.org/whl/xpu
(and current Intel GPU drivers). See README.md for whether that's set up
on this machine - if not, everything here still works, it just resolves
to "cpu".
"""

def select_device():
    # Imported lazily, not at module scope: detector.py imports this module
    # unconditionally (not inside its build_detector() try/except), so a
    # top-level `import torch` here would crash that import - and therefore
    # kill the HOG fallback too - on any machine/deploy that doesn't have
    # torch installed at all. select_device() itself is only ever called
    # from inside YOLOPersonDetector.__init__ and ppe_classifier's ViT
    # loading path, both of which are already guarded by their own
    # try/except, so torch being missing there still degrades gracefully.
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"
