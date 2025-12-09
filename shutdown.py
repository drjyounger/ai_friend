#!/usr/bin/env python3
"""
Companion Shutdown Script
=======================
Cleanly shuts down all Companion processes and releases the webcam.

Usage:
    python shutdown.py
"""

import subprocess
import time
import sys


def kill_companion():
    """Kill all Companion-related Python processes."""
    print("🛑 Shutting down Companion...")
    
    # Find Python processes running main.py or companion
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py|python.*companion"],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().split('\n')
        pids = [p for p in pids if p]  # Remove empty strings
        
        if pids:
            print(f"   Found {len(pids)} process(es) to kill...")
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False)
                    print(f"   Killed PID {pid}")
                except Exception:
                    pass
        else:
            print("   No Companion processes found.")
            
    except Exception as e:
        print(f"   Warning: {e}")
    
    # Also try killing by name pattern
    subprocess.run(["pkill", "-9", "-f", "python.*main.py"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "Python.*main.py"], capture_output=True)
    
    time.sleep(0.5)


def release_webcam():
    """Release any webcam connections."""
    print("📷 Releasing webcam...")
    
    try:
        import cv2
        # Try to open and immediately release cameras
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
        print("   Webcam released.")
    except ImportError:
        print("   OpenCV not available (webcam may already be released)")
    except Exception as e:
        print(f"   Note: {e}")


def verify_shutdown():
    """Verify everything is shut down."""
    print("🔍 Verifying shutdown...")
    
    result = subprocess.run(
        ["pgrep", "-f", "python.*main.py"],
        capture_output=True,
        text=True
    )
    
    if result.stdout.strip():
        print("   ⚠️  Some processes may still be running")
        return False
    else:
        print("   ✅ All clear!")
        return True


def main():
    print()
    print("=" * 50)
    print("  COMPANION SHUTDOWN")
    print("=" * 50)
    print()
    
    kill_companion()
    release_webcam()
    
    print()
    success = verify_shutdown()
    
    print()
    if success:
        print("👋 Companion has been shut down completely.")
    else:
        print("⚠️  Manual intervention may be needed.")
        print("   Try: pkill -9 -f python")
    
    print()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

