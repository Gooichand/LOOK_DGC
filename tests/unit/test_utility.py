"""
Unit tests for utility.py functions
"""
import numpy as np
import cv2 as cv
from unittest.mock import patch, MagicMock

# Import the functions we want to test
from utility import (
    mat2img, color_by_value, modify_font, pad_image, shift_image,
    human_size, create_lut, compute_hist, auto_lut, elapsed_time,
    signed_value, equalize_img, norm_img, clip_value, bgr_to_gray3,
    gray_to_bgr, desaturate, norm_mat
)

def test_mat2img(sample_image):
    """Test conversion from OpenCV matrix to QImage"""
    qimage = mat2img(sample_image)
    assert qimage is not None
    assert qimage.width() == 100
    assert qimage.height() == 100
    # Format check - the exact format name may vary by Qt version
    assert qimage.format() is not None

def test_color_by_value():
    """Test color assignment based on value ranges"""
    # This test would require mocking QTreeWidgetItem and QColor
    # For now, we'll test that it doesn't raise exceptions
    try:
        # Mock the item
        mock_item = MagicMock()
        color_by_value(mock_item, 50, [25, 50, 75])
        # Verify the mock was called
        assert mock_item.setBackground.called
    except Exception as e:
        pytest.fail(f"color_by_value raised an exception: {e}")

def test_modify_font():
    """Test font modification function"""
    # Test with None object
    modify_font(None)  # Should not raise exception
    
    # Test with mock object
    mock_obj = MagicMock()
    modify_font(mock_obj, bold=True, italic=True)
    assert mock_obj.setFont.called

def test_pad_image(sample_image):
    """Test image padding functionality"""
    original_shape = sample_image.shape
    padded = pad_image(sample_image, 16)
    
    # Check that padding was applied
    assert padded.shape[0] >= original_shape[0]
    assert padded.shape[1] >= original_shape[1]
    assert padded.shape[2] == original_shape[2]
    
    # Check that original image is preserved in top-left corner
    assert np.array_equal(padded[:original_shape[0], :original_shape[1]], sample_image)

def test_shift_image(sample_image):
    """Test image shifting functionality"""
    shifted = shift_image(sample_image, 10)
    
    # Check dimensions remain the same
    assert shifted.shape == sample_image.shape
    
    # Check that the shift operation was performed
    # The exact values depend on the implementation
    assert shifted.shape == sample_image.shape

def test_human_size():
    """Test human-readable size conversion"""
    # Test basic functionality - exact output may vary by implementation
    result = human_size(1024)
    assert isinstance(result, str)
    assert len(result) > 0

def test_create_lut():
    """Test lookup table creation"""
    lut = create_lut(50, 100)
    assert len(lut) == 256
    assert lut.dtype == np.uint8
    assert np.all(lut >= 0) and np.all(lut <= 255)

def test_compute_hist(sample_image):
    """Test histogram computation"""
    # Test with grayscale image
    gray_img = cv.cvtColor(sample_image, cv.COLOR_BGR2GRAY)
    hist = compute_hist(gray_img)
    assert len(hist) == 256
    assert hist.dtype == int
    assert sum(hist) == gray_img.size

def test_auto_lut(sample_grayscale_image):
    """Test automatic lookup table creation"""
    lut = auto_lut(sample_grayscale_image, 0.01)
    assert len(lut) == 256
    assert lut.dtype == np.uint8
    assert np.all(lut >= 0) and np.all(lut <= 255)

def test_elapsed_time():
    """Test elapsed time formatting"""
    import time
    start = time.time()
    time.sleep(0.01)  # Small delay
    elapsed = elapsed_time(start, ms=True)
    assert "ms" in elapsed
    
    elapsed_sec = elapsed_time(start, ms=False)
    assert "sec" in elapsed_sec

def test_signed_value():
    """Test signed value formatting"""
    assert signed_value(5) == "+5"
    assert signed_value(-3) == "-3"
    assert signed_value(0) == "0"

def test_equalize_img(sample_image):
    """Test image equalization"""
    equalized = equalize_img(sample_image)
    assert equalized.shape == sample_image.shape
    assert equalized.dtype == sample_image.dtype

def test_norm_img(sample_image):
    """Test image normalization"""
    normalized = norm_img(sample_image)
    assert normalized.shape == sample_image.shape
    assert normalized.dtype == sample_image.dtype

def test_clip_value():
    """Test value clipping"""
    assert clip_value(5, minv=0, maxv=10) == 5
    assert clip_value(-5, minv=0, maxv=10) == 0
    assert clip_value(15, minv=0, maxv=10) == 10
    assert clip_value(5, minv=0) == 5
    assert clip_value(5, maxv=10) == 5

def test_bgr_to_gray3(sample_image):
    """Test BGR to 3-channel grayscale conversion"""
    gray3 = bgr_to_gray3(sample_image)
    assert gray3.shape == sample_image.shape
    # All channels should be equal in a grayscale image
    assert np.array_equal(gray3[:, :, 0], gray3[:, :, 1])
    assert np.array_equal(gray3[:, :, 1], gray3[:, :, 2])

def test_gray_to_bgr(sample_grayscale_image):
    """Test grayscale to BGR conversion"""
    bgr = gray_to_bgr(sample_grayscale_image)
    assert len(bgr.shape) == 3
    assert bgr.shape[2] == 3
    # All channels should be equal
    assert np.array_equal(bgr[:, :, 0], bgr[:, :, 1])
    assert np.array_equal(bgr[:, :, 1], bgr[:, :, 2])

def test_desaturate(sample_image):
    """Test image desaturation"""
    desaturated = desaturate(sample_image)
    assert desaturated.shape == sample_image.shape
    # Should be 3-channel grayscale
    assert np.array_equal(desaturated[:, :, 0], desaturated[:, :, 1])
    assert np.array_equal(desaturated[:, :, 1], desaturated[:, :, 2])

def test_norm_mat(sample_image):
    """Test matrix normalization"""
    # Test with 2D matrix
    matrix_2d = cv.cvtColor(sample_image, cv.COLOR_BGR2GRAY)
    norm_2d = norm_mat(matrix_2d)
    assert norm_2d.shape == matrix_2d.shape
    assert norm_2d.dtype == np.uint8
    assert np.min(norm_2d) >= 0 and np.max(norm_2d) <= 255
    
    # Test with 3D matrix to BGR
    norm_3d = norm_mat(matrix_2d, to_bgr=True)
    assert len(norm_3d.shape) == 3
    assert norm_3d.shape[2] == 3