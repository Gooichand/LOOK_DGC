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
import importlib.util

def check_internet():
    """Check if internet connection is available"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def check_dependencies():
    """Check if all required Python dependencies are installed using the validator"""
    try:
        # Try to import the validator
        validator_path = os.path.join(os.path.dirname(__file__), 'validate_deps.py')
        if os.path.exists(validator_path):
            import validate_deps
            return validate_deps.validate_dependencies(detailed=True)
        else:
            # Fallback to basic checking
            return check_dependencies_basic()
    except ImportError:
        # Fallback to basic checking if validator can't be imported
        return check_dependencies_basic()

def check_dependencies_basic():
    """Basic dependency checking as fallback"""
    print("Checking Python dependencies...")
    
    # Required packages and their import names
    required_packages = {
        'PySide6': 'PySide6',
        'opencv-contrib-python-headless': 'cv2',
        'numpy': 'numpy',
        'Pillow': 'PIL',
        'matplotlib': 'matplotlib',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'lxml': 'lxml',
        'python-magic': 'magic',
        'rawpy': 'rawpy',
        'sewar': 'sewar',
        'PyWavelets': 'pywt',
        'astor': 'astor',
        'concurrent-iterator': 'concurrent_iterator',
        'keras-applications': 'keras_applications',
        'xgboost': 'xgboost',
        'reportlab': 'reportlab'
    }
    
    # Try to import tensorflow (optional but important for some features)
    try:
        import tensorflow as tf
        tensorflow_available = True
        print("✓ TensorFlow available")
    except ImportError:
        tensorflow_available = False
        print("⚠ TensorFlow not found - some advanced features will be disabled")
    
    missing_packages = []
    available_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            # Try to import the module
            importlib.import_module(import_name)
            available_packages.append(package_name)
            print(f"✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} (missing)")
    
    all_good = len(missing_packages) == 0
    
    if not all_good:
        print(f"\n❌ Missing {len(missing_packages)} required dependencies:")
        for package_name in missing_packages:
            print(f"   - {package_name}")
        
        print("\n🔧 Installation options:")
        print("1. Automatic installation (recommended):")
        print("   python check_deps.py")
        print("\n2. Manual installation:")
        print("   pip install -r gui/requirements.txt")
        print("\n3. Install individual packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    else:
        print(f"\n✅ All {len(available_packages)} required dependencies are installed")
        if not tensorflow_available:
            print("⚠ Note: TensorFlow is recommended for full functionality")
    
    return all_good, available_packages, missing_packages

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
    
    # Check dependencies before proceeding
    deps_ok, available, missing = check_dependencies()
    if not deps_ok:
        print("\n⚠ Cannot launch LOOK-DGC due to missing dependencies.")
        choice = input("\nWould you like to try launching anyway? (y/N): ").strip().lower()
        if choice != 'y':
            print("Launch aborted. Please install the missing dependencies.")
            input("Press Enter to exit...")
            return
        else:
            print("⚠ Proceeding with launch despite missing dependencies...")
            print("Some features may not work correctly.")
    
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