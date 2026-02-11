"""
pytest configuration and fixtures for LOOK-DGC tests
"""
import os
import sys
import pytest
import cv2 as cv
import numpy as np

# Add the gui directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gui'))

# Test image fixtures
@pytest.fixture
def sample_image():
    """Create a simple test image"""
    # Create a 100x100 RGB image with a gradient
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            image[i, j] = [i * 2, j * 2, (i + j)]
    return image

@pytest.fixture
def sample_grayscale_image():
    """Create a simple grayscale test image"""
    image = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            image[i, j] = (i + j) % 256
    return image

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary image file for testing"""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    file_path = tmp_path / "test_image.jpg"
    cv.imwrite(str(file_path), image)
    return str(file_path)

@pytest.fixture
def mock_qt_app(qtbot):
    """Create a mock Qt application for testing GUI components"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app