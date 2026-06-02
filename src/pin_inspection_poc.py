import sys
import os

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sandbox.pin_inspection_poc import inspect_pins

if __name__ == '__main__':
    print("[Compatibility Stub] Forwarding execution from pin_inspection_poc.py to src/sandbox/pin_inspection_poc.py...")
    inspect_pins()
