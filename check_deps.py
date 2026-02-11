import sys
import subprocess
import importlib

def check_and_install_packages():
    required_packages = [
        'astor',
        'concurrent-iterator',
        'keras-applications',
        'lxml',
        'matplotlib',
        'opencv-contrib-python-headless',
        'pandas',
        'pyside6',
        'python-magic',
        'rawpy',
        'scikit-learn',
        'sewar',
        'tensorflow',
        'xgboost',
        'pillow',
        'PyWavelets'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_').split('==')[0])
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        
        # Install packages individually to handle failures better
        installed_successfully = []
        failed_packages = []
        
        for package in missing_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                installed_successfully.append(package)
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                failed_packages.append(package)
                print(f"✗ Failed to install {package}")
        
        if installed_successfully:
            print(f"Successfully installed: {', '.join(installed_successfully)}")
        
        if failed_packages:
            print(f"Warning: Failed to install: {', '.join(failed_packages)}")
            print("You may need to install these packages manually.")
            # Don't exit with error, just warn
        else:
            print("All dependencies installed successfully!")
    else:
        print("All dependencies are already installed.")

if __name__ == "__main__":
    check_and_install_packages()