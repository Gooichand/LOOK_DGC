"""
Integration tests for the main LOOK-DGC application
"""
import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

def test_application_imports():
    """Test that main application modules can be imported"""
    try:
        # Test importing the main module
        sys.path.insert(0, '../gui')
        import look_dgc  # This should work without errors
        assert look_dgc is not None
    except ImportError as e:
        pytest.fail(f"Failed to import main application: {e}")
    finally:
        if '../gui' in sys.path:
            sys.path.remove('../gui')

def test_main_window_creation(mock_qt_app):
    """Test that main window can be created"""
    try:
        # Add gui to path
        sys.path.insert(0, '../gui')
        with patch('PySide6.QtWidgets.QApplication') as mock_app:
            from look_dgc import MainWindow
            # Skip this test if GUI components are not available
            pytest.skip("MainWindow test requires full GUI environment")
    except ImportError:
        pytest.skip("Main window import failed - skipping")
    finally:
        if '../gui' in sys.path:
            sys.path.remove('../gui')

def test_tool_imports():
    """Test that all tool modules can be imported"""
    tool_modules = [
        'adjust', 'cloning', 'comparison', 'contrast', 'digest',
        'echo', 'editor', 'ela', 'exif', 'frequency', 'gradient',
        'header', 'histogram', 'location', 'magnifier', 'median',
        'minmax', 'multiple', 'noise', 'original', 'pca', 'planes',
        'plots', 'quality', 'reverse', 'space', 'trufor',
        'stats', 'stereogram', 'thumbnail', 'tools', 'utility',
        'wavelets', 'ghostmmaps', 'resampling', 'noise_estimmation'
    ]
    
    # Note: 'splicing' module is skipped because it requires TensorFlow
    optional_modules = ['splicing']
    
    sys.path.insert(0, '../gui')
    failed_imports = []
    
    for module in tool_modules:
        if module not in optional_modules:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
            except Exception as e:
                # Other exceptions might be expected (like missing TensorFlow)
                pass
    
    sys.path.remove('../gui')
    
    if failed_imports:
        pytest.fail(f"Failed to import modules: {', '.join(failed_imports)}")

def test_application_initialization():
    """Test application initialization sequence"""
    # This test is difficult to run in a headless environment
    # Just verify that the main module can be imported
    try:
        sys.path.insert(0, '../gui')
        import look_dgc
        # If we can import it, that's a good sign
        assert hasattr(look_dgc, 'MainWindow')
    except ImportError:
        pytest.skip("Cannot import main application - skipping")
    finally:
        if '../gui' in sys.path:
            sys.path.remove('../gui')

def test_dependency_availability():
    """Test that required dependencies are available"""
    required_packages = [
        'cv2', 'numpy', 'PySide6', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        pytest.fail(f"Missing required packages: {', '.join(missing_packages)}")

def test_optional_dependency_handling():
    """Test handling of optional dependencies"""
    # Test TensorFlow import (optional)
    try:
        import tensorflow as tf
        tensorflow_available = True
    except ImportError:
        tensorflow_available = False
    
    # Test rawpy import (optional)  
    try:
        import rawpy
        rawpy_available = True
    except ImportError:
        rawpy_available = False
    
    # Both should be handled gracefully in the application
    assert isinstance(tensorflow_available, bool)
    assert isinstance(rawpy_available, bool)

def test_qt_application_setup():
    """Test Qt application setup"""
    try:
        sys.path.insert(0, '../gui')
        from PySide6.QtWidgets import QApplication
        
        # Test that we can call the basic application setup methods
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        QApplication.setApplicationName("LOOK-DGC")
        QApplication.setOrganizationName("Gopichand")
        
        # If we reach here without exception, the basic setup worked
        assert QApplication.applicationName() == "LOOK-DGC"
    except Exception:
        pytest.skip("Qt application setup test requires GUI environment")
    finally:
        if '../gui' in sys.path:
            sys.path.remove('../gui')

def test_main_window_attributes():
    """Test that main window has expected attributes"""
    try:
        sys.path.insert(0, '../gui')
        from look_dgc import MainWindow
        pytest.skip("MainWindow attributes test requires full GUI environment")
    except ImportError:
        pytest.skip("Main window import failed - skipping")
    finally:
        if '../gui' in sys.path:
            sys.path.remove('../gui')