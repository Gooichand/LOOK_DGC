#!/usr/bin/env python3
"""
LOOK-DGC Launcher
Developed by: Gopichand
Project: LOOK-DGC - Digital Image Forensics Toolkit
"""

import sys
import os
import subprocess
import socket

def check_internet():
    """Check if internet connection is available"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def main():
    print("="*50)
    print("  LOOK-DGC - Digital Image Forensics Toolkit")
    print("  Developed by: Gopichand")
    print("="*50)
    
    # Check network status
    online = check_internet()
    if online:
        print("  🌐 Status: Online - Full features available")
        print("  📡 Hex editor: Online (HexEd.it) + Offline built-in viewer ready")
    else:
        print("  📡 Status: Offline - Limited features available")
        print("  💾 Hex editor: Offline built-in viewer ready")
    print()
    
    # Change to the gui directory
    gui_dir = os.path.join(os.path.dirname(__file__), 'gui')
    if not os.path.exists(gui_dir):
        print(f"Error: GUI directory not found at {gui_dir}")
        input("Press Enter to exit...")
        return
    
    os.chdir(gui_dir)
    
    # Launch LOOK-DGC
    try:
        print("Starting LOOK-DGC...")
        subprocess.run([sys.executable, 'look-dgc.py'], check=True)
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except FileNotFoundError:
        print("Error: look-dgc.py not found in gui directory")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"Error launching LOOK-DGC: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()