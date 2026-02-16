#!/usr/bin/env python3
"""Screen recorder for Simpleem POC demo.

Usage:
    python3 record_demo.py          # Records screen 0 (main display)
    python3 record_demo.py --screen 1  # Records screen 1 (external display)

Press Ctrl+C to stop recording. The video is saved to demo_recording.mp4
"""

import subprocess
import sys
import signal
import os
import webbrowser
import time

DEMO_URL = "http://localhost:5173"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "demo_recording.mp4")

# Parse screen index from args
screen_index = 0
if "--screen" in sys.argv:
    idx = sys.argv.index("--screen")
    if idx + 1 < len(sys.argv):
        screen_index = int(sys.argv[idx + 1])

# AVFoundation screen device index = 3 + screen_index
device_index = 3 + screen_index

print("=" * 60)
print("  Simpleem POC - Demo Screen Recorder")
print("=" * 60)
print()
print(f"  Recording: Screen {screen_index} (device index {device_index})")
print(f"  Output:    {OUTPUT_FILE}")
print(f"  Dashboard: {DEMO_URL}")
print()
print("  Opening dashboard in browser...")
time.sleep(1)
webbrowser.open(DEMO_URL)
time.sleep(2)

print()
print("  Recording starts in 3 seconds...")
print("  >>> Press Ctrl+C to stop recording <<<")
print()
time.sleep(3)

# FFmpeg command for macOS screen capture
cmd = [
    "ffmpeg",
    "-y",                          # Overwrite output
    "-f", "avfoundation",          # macOS screen capture
    "-framerate", "30",            # 30 FPS
    "-capture_cursor", "1",        # Show mouse cursor
    "-i", f"{device_index}:none",  # Screen device : no audio
    "-vf", "scale=-2:1080",        # Scale to 1080p height
    "-c:v", "libx264",             # H.264 codec
    "-preset", "ultrafast",        # Fast encoding for real-time
    "-crf", "23",                  # Good quality
    "-pix_fmt", "yuv420p",         # Compatible pixel format
    OUTPUT_FILE,
]

print("  [RECORDING] Move your mouse to the dashboard and demo!")
print()

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)


def stop_recording(signum=None, frame=None):
    print()
    print("  Stopping recording...")
    process.send_signal(signal.SIGINT)
    process.wait(timeout=10)
    print(f"  Recording saved to: {OUTPUT_FILE}")
    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    print()
    print("  Done! Share this file with your team.")
    sys.exit(0)


signal.signal(signal.SIGINT, stop_recording)
signal.signal(signal.SIGTERM, stop_recording)

# Wait for ffmpeg to finish (or Ctrl+C)
process.wait()
