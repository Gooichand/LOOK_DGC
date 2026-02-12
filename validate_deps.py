#!/usr/bin/env python3
"""
LOOK-DGC Dependency Validator
Validates that all required Python packages are installed before launching the application.
This script provides detailed information about missing dependencies and installation guidance.
"""

import sys
import os
import importlib.util
from typing import Dict, List, Tuple

def get_required_packages() -> Dict[str, str]:
    """Return dictionary of required packages and their import names"""
    return {
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

def check_package(package_name: str, import_name: str) -> Tuple[bool, str]:
    """
    Check if a package can be imported.
    
    Returns:
        Tuple[bool, str]: (is_available, status_message)
    """
    try:
        importlib.import_module(import_name)
        return True, f"✓ {package_name} is available"
    except ImportError as e:
        return False, f"✗ {package_name} is missing: {str(e)}"

def check_tensorflow() -> Tuple[bool, str]:
    """Check TensorFlow availability (optional but recommended)"""
    try:
        import tensorflow as tf
        version = tf.__version__
        return True, f"✓ TensorFlow {version} is available"
    except ImportError:
        return False, "⚠ TensorFlow not found - some advanced features will be disabled"

def validate_dependencies(detailed: bool = True) -> Tuple[bool, List[str], List[str]]:
    """
    Validate all required dependencies.
    
    Args:
        detailed: If True, print detailed output to console
        
    Returns:
        Tuple[bool, List[str], List[str]]: (all_good, available_packages, missing_packages)
    """
    if detailed:
        print("🔍 Validating LOOK-DGC dependencies...")
        print("=" * 50)
    
    required_packages = get_required_packages()
    available_packages = []
    missing_packages = []
    
    # Check TensorFlow
    tf_available, tf_message = check_tensorflow()
    if detailed:
        print(tf_message)
    
    # Check all required packages
    if detailed:
        print("\n📦 Required packages:")
    
    for package_name, import_name in required_packages.items():
        is_available, message = check_package(package_name, import_name)
        if detailed:
            print(f"  {message}")
        
        if is_available:
            available_packages.append(package_name)
        else:
            missing_packages.append(package_name)
    
    # Summary
    if detailed:
        print("\n" + "=" * 50)
        print(f"✅ Available: {len(available_packages)}/{len(required_packages)}")
        if missing_packages:
            print(f"❌ Missing: {len(missing_packages)}")
            for pkg in missing_packages:
                print(f"   - {pkg}")
        else:
            print("✅ All required packages are installed!")
        
        if not tf_available:
            print("⚠ TensorFlow is recommended for full functionality")
    
    return len(missing_packages) == 0, available_packages, missing_packages

def get_installation_guidance(missing_packages: List[str]) -> str:
    """Generate installation guidance for missing packages"""
    if not missing_packages:
        return "All dependencies are satisfied."
    
    guidance = "🔧 Installation Options:\n\n"
    
    # Option 1: Use existing check_deps.py
    guidance += "1. Automatic installation (recommended):\n"
    guidance += "   python check_deps.py\n\n"
    
    # Option 2: Install from requirements.txt
    guidance += "2. Manual installation via requirements.txt:\n"
    guidance += "   pip install -r gui/requirements.txt\n\n"
    
    # Option 3: Install individual packages
    if missing_packages:
        guidance += "3. Install individual missing packages:\n"
        package_list = " ".join(missing_packages)
        guidance += f"   pip install {package_list}\n\n"
    
    # Option 4: Install all packages
    all_packages = list(get_required_packages().keys())
    all_package_list = " ".join(all_packages)
    guidance += "4. Install all required packages:\n"
    guidance += f"   pip install {all_package_list}\n"
    
    return guidance

def main():
    """Main function for standalone execution"""
    print("LOOK-DGC Dependency Validator")
    print("=" * 40)
    
    # Validate dependencies
    all_good, available, missing = validate_dependencies(detailed=True)
    
    if missing:
        print("\n" + "=" * 50)
        print(get_installation_guidance(missing))
        
        # Check if check_deps.py exists and suggest using it
        check_deps_path = os.path.join(os.path.dirname(__file__), 'check_deps.py')
        if os.path.exists(check_deps_path):
            print("\n💡 Quick fix:")
            print("   python check_deps.py")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())