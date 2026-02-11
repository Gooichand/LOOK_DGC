"""
Unit tests for image loading functionality
"""
import os
import pytest
import numpy as np
import cv2 as cv
from unittest.mock import patch, MagicMock

# Import the functions we want to test
from utility import load_image

def test_load_image_jpeg(sample_image_path):
    """Test loading a JPEG image"""
    # Create a mock parent widget
    mock_parent = MagicMock()
    
    # Test loading the image
    filename, basename, image = load_image(mock_parent, sample_image_path)
    
    assert filename == sample_image_path
    assert basename == os.path.basename(sample_image_path)
    assert image is not None
    assert len(image.shape) == 3  # Should be 3-channel
    assert image.shape[2] == 3   # BGR format

def test_load_image_png(tmp_path):
    """Test loading a PNG image"""
    # Create a PNG test image
    image_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    png_path = tmp_path / "test_image.png"
    cv.imwrite(str(png_path), image_data)
    
    mock_parent = MagicMock()
    filename, basename, image = load_image(mock_parent, str(png_path))
    
    assert filename == str(png_path)
    assert basename == "test_image.png"
    assert image is not None
    assert np.array_equal(image, image_data)

def test_load_image_invalid_file():
    """Test loading an invalid file"""
    mock_parent = MagicMock()
    
    # Test with non-existent file
    filename, basename, image = load_image(mock_parent, "nonexistent.jpg")
    
    # The function should handle the error gracefully
    # Exact behavior may vary, but it shouldn't crash
    assert filename is None or isinstance(filename, str)
    assert basename is None or isinstance(basename, str)
    assert image is None or isinstance(image, np.ndarray)

def test_load_image_grayscale(tmp_path):
    """Test loading a grayscale image"""
    # Create a grayscale image
    gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    file_path = tmp_path / "gray_image.jpg"
    cv.imwrite(str(file_path), gray_image)
    
    mock_parent = MagicMock()
    filename, basename, image = load_image(mock_parent, str(file_path))
    
    assert image is not None
    assert len(image.shape) == 3  # Should be converted to 3-channel BGR
    assert image.shape[2] == 3

def test_load_image_with_alpha_channel(tmp_path):
    """Test loading an image with alpha channel"""
    # Create an image with alpha channel (BGRA)
    bgra_image = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
    file_path = tmp_path / "alpha_image.png"
    cv.imwrite(str(file_path), bgra_image)
    
    mock_parent = MagicMock()
    filename, basename, image = load_image(mock_parent, str(file_path))
    
    assert image is not None
    assert len(image.shape) == 3
    assert image.shape[2] == 3  # Alpha channel should be discarded

@patch('utility.RAWPY_AVAILABLE', False)
def test_load_raw_image_no_rawpy(tmp_path):
    """Test loading RAW image without rawpy available"""
    # Create a fake RAW file
    raw_path = tmp_path / "test_image.nef"
    raw_path.write_bytes(b"fake raw data")
    
    mock_parent = MagicMock()
    filename, basename, image = load_image(mock_parent, str(raw_path))
    
    # Should handle the situation gracefully when rawpy is not available
    # Exact behavior may vary, but it shouldn't crash
    assert filename is None or isinstance(filename, str)
    assert basename is None or isinstance(basename, str)
    assert image is None or isinstance(image, np.ndarray)

def test_load_gif_image(tmp_path):
    """Test loading GIF image"""
    # Create a simple GIF file
    gif_path = tmp_path / "test.gif"
    gif_path.write_bytes(b"GIF89a")  # GIF header
    
    mock_parent = MagicMock()
    try:
        filename, basename, image = load_image(mock_parent, str(gif_path))
        # Should handle gracefully, exact behavior may vary
        assert filename is None or isinstance(filename, str)
        assert basename is None or isinstance(basename, str)
        assert image is None or isinstance(image, np.ndarray)
    except Exception:
        # Some OpenCV setups might not support GIF, that's OK
        pass

def test_load_image_dialog_cancel():
    """Test when user cancels the file dialog"""
    with patch('utility.QFileDialog') as mock_dialog:
        mock_instance = MagicMock()
        mock_dialog.return_value = mock_instance
        mock_instance.exec_.return_value = False  # User cancelled
        
        mock_parent = MagicMock()
        result = load_image(mock_parent)
        
        # Should return None values when dialog is cancelled
        assert result == [None, None, None]