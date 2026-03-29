#!/usr/bin/env python3
"""
Verify Advanced AI Installation
Check if YOLOv11, Vision Transformers, and Hugging Face are ready
"""

import sys

def check_installation():
    print("=" * 70)
    print("ADVANCED AI INSTALLATION VERIFICATION")
    print("=" * 70)
    
    checks = []
    
    # Check PyTorch
    try:
        import torch
        print(f"\n✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ️  Running on CPU (GPU recommended but not required)")
        checks.append(True)
    except ImportError:
        print("\n❌ PyTorch not installed")
        print("   Install: pip install torch torchvision")
        checks.append(False)
    
    # Check Ultralytics (YOLO)
    try:
        from ultralytics import YOLO
        print(f"\n✅ Ultralytics (YOLO) installed")
        print("   Can use: YOLOv8, YOLOv11")
        checks.append(True)
    except ImportError:
        print("\n❌ Ultralytics not installed")
        print("   Install: pip install ultralytics")
        checks.append(False)
    
    # Check Transformers
    try:
        import transformers
        print(f"\n✅ Hugging Face Transformers: {transformers.__version__}")
        print("   Can use: DETR, ViT, and 100,000+ models")
        checks.append(True)
    except ImportError:
        print("\n❌ Transformers not installed")
        print("   Install: pip install transformers accelerate")
        checks.append(False)
    
    # Check TIMM
    try:
        import timm
        print(f"\n✅ TIMM (Vision Models): {timm.__version__}")
        checks.append(True)
    except ImportError:
        print("\n❌ TIMM not installed (optional)")
        print("   Install: pip install timm")
        checks.append(False)
    
    # Check OpenCV (should already be installed)
    try:
        import cv2
        print(f"\n✅ OpenCV: {cv2.__version__}")
        checks.append(True)
    except ImportError:
        print("\n❌ OpenCV not installed")
        print("   Install: pip install opencv-python")
        checks.append(False)
    
    # Check NumPy
    try:
        import numpy as np
        print(f"\n✅ NumPy: {np.__version__}")
        checks.append(True)
    except ImportError:
        print("\n❌ NumPy not installed")
        checks.append(False)
    
    print("\n" + "=" * 70)
    
    if all(checks):
        print("✅ ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("=" * 70)
        print("\n🎉 You're ready to use advanced AI models!")
        print("\nNext steps:")
        print("1. Run: python hybrid_detector_advanced.py")
        print("2. Read: HYBRID_AI_COMPLETE_GUIDE.md")
        print("3. Experiment with different models!")
        return True
    else:
        print("❌ SOME PACKAGES MISSING")
        print("=" * 70)
        print("\n📝 Installation command:")
        print("pip install torch torchvision ultralytics transformers accelerate timm")
        print("\nOr on some systems:")
        print("pip install torch torchvision ultralytics transformers accelerate timm --break-system-packages")
        return False

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)