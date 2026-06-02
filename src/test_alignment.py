import sys
import os

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sandbox.test_alignment import align_images

if __name__ == '__main__':
    print("[Compatibility Stub] Forwarding execution from test_alignment.py to src/sandbox/test_alignment.py...")
    align_images()
