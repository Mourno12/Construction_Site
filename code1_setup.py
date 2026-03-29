#!/usr/bin/env python3
"""
Code 1: Installation and Setup Verification
Checks all required libraries for person detection project
"""

import sys

def check_installation():
    print("=" * 60)
    print("INSTALLATION VERIFICATION")
    print("=" * 60)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    
    # Check PIL/Pillow
    try:
        from PIL import Image
        print(f"✓ PIL/Pillow is installed")
    except ImportError:
        print("✗ PIL/Pillow not installed")
        return False
    
    # Check numpy (required by OpenCV)
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    print("\n" + "=" * 60)
    print("ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
    print("=" * 60)
    print("\nReady to proceed with person detection.")
    return True

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)