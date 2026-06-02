import sys
import os

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.camera import main

if __name__ == '__main__':
    print("[Compatibility Stub] Forwarding execution from camera_stream_v2.py to src/core/camera.py...")
    main()
