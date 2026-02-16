import os
import sys
from time import time
from math import log10

try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False

import cv2 as cv
import numpy as np
from PySide6.QtCore import QSettings, QFileInfo, Signal, Qt, QMimeDatabase
from PySide6.QtGui import QImage, QFontDatabase, QColor, QBrush
from PySide6.QtWidgets import (
    QLabel,
    QTreeWidgetItem,
    QFileDialog,
    QMessageBox,
    QWidget,
    QSlider,
    QSpinBox,
    QHBoxLayout,
)


def mat2img(cvmat):
    height, width, channels = cvmat.shape
    return QImage(cvmat.data, width, height, 3 * width, QImage.Format_BGR888)


def color_by_value(item, value, splits):
    if value < splits[0]:
        hue = 0
    elif value < splits[1]:
        hue = 30
    elif value < splits[2]:
        hue = 60
    else:
        hue = 90
    item.setBackground(QBrush(QColor.fromHsv(hue, 96, 255)))


def modify_font(obj, bold=False, italic=False, underline=False, mono=False):
    if obj is None:
        return
    if mono:
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    else:
        if type(obj) is QTreeWidgetItem:
            font = obj.font(0)
        else:
            font = obj.font()
    font.setBold(bold)
    font.setItalic(italic)
    font.setUnderline(underline)
    if type(obj) is QTreeWidgetItem:
        obj.setFont(0, font)
    else:
        obj.setFont(font)


def pad_image(image, bsize, reflect=False):
    rows, cols = image.shape[:2]
    top = left = 0
    bottom = bsize - rows % bsize
    right = bsize - cols % bsize
    border = cv.BORDER_CONSTANT if not reflect else cv.BORDER_REFLECT_101
    padded = cv.copyMakeBorder(image, top, bottom, left, right, border)
    return padded


def shift_image(image, bsize):
    rows, cols = image.shape[:2]
    shifted = np.zeros_like(image)
    shifted[: rows - bsize, : cols - bsize] = image[bsize:, bsize:]
    return shifted


def human_size(total, binary=False, suffix="B"):
    units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    if binary:
        units = [unit + "i" for unit in units]
        factor = 1024.0
    else:
        factor = 1000.0
    for unit in units:
        if abs(total) < factor:
            return f"{total:3.1f} {unit}{suffix}"
        total /= factor
    return f"{total:.1f} {units[-1]}{suffix}"


def create_lut(low, high):
    if low >= 0:
        p1 = (+low, 0)
    else:
        p1 = (0, -low)
    if high >= 0:
        p2 = (255 - high, 255)
    else:
        p2 = (255, 255 + high)
    if p1[0] == p2[0]:
        return np.full(256, 255, np.uint8)
    lut = [
        (x * (p1[1] - p2[1]) + p1[0] * p2[1] - p1[1] * p2[0]) / (p1[0] - p2[0])
        for x in range(256)
    ]
    return np.clip(np.array(lut), 0, 255).astype(np.uint8)


def compute_hist(image, normalize=False):
    hist = np.array(
        [h[0] for h in cv.calcHist([image], [0], None, [256], [0, 256])], int
    )
    return hist / image.size if normalize else hist


def auto_lut(image, centile):
    hist = compute_hist(image, normalize=True)
    if centile == 0:
        nonzero = np.nonzero(hist)[0]
        low = nonzero[0]
        high = nonzero[-1]
    else:
        low_sum = high_sum = 0
        low = 0
        high = 255
        for i, h in enumerate(hist):
            low_sum += h
            if low_sum >= centile:
                low = i
                break
        for i, h in enumerate(np.flip(hist)):
            high_sum += h
            if high_sum >= centile:
                high = i
                break
    return create_lut(low, high)


def elapsed_time(start, ms=True):
    elapsed = time() - start
    if ms:
        return f"{int(np.round(elapsed * 1000))} ms"
    return f"{elapsed:.2f} sec"


def signed_value(value):
    return f"{'+' if value > 0 else ''}{value}"


def equalize_img(image):
    return cv.merge([cv.equalizeHist(c) for c in cv.split(image)])


def norm_img(image):
    return cv.merge([norm_mat(c) for c in cv.split(image)])


def clip_value(value, minv=None, maxv=None):
    if minv is not None:
        value = max(value, minv)
    if maxv is not None:
        value = min(value, maxv)
    return value


def bgr_to_gray3(image):
    return cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)


def gray_to_bgr(image):
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def load_image(parent, filename=None):
    nothing = [None] * 3
    settings = QSettings()
    mime_filters = [
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/gif",
        "image/bmp",
        "image/webp",
        "image/x-portable-pixmap",
        "image/x-portable-graymap",
        "image/x-portable-bitmap",
        "image/x-nikon-nef",
        "image/x-fuji-raf",
        "image/x-canon-cr2",
        "image/x-adobe-dng",
        "image/x-sony-arw",
        "image/x-kodak-dcr",
        "image/x-minolta-mrw",
        "image/x-pentax-pef",
        "image/x-canon-crw",
        "image/x-sony-sr2",
        "image/x-olympus-orf",
        "image/x-panasonic-raw",
    ]
    mime_db = QMimeDatabase()
    mime_patterns = [
        mime_db.mimeTypeForName(mime).globPatterns() for mime in mime_filters
    ]
    all_formats = f"Supported formats ({' '.join([item for sub in mime_patterns for item in sub])})"
    raw_exts = [p[-3:] for p in mime_patterns][-12:]
    if filename is None:
        dialog = QFileDialog(
            parent, parent.tr("Load image"), settings.value("load_folder")
        )
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        dialog.setMimeTypeFilters(mime_filters)
        name_filters = dialog.nameFilters()
        dialog.setNameFilters(name_filters + [all_formats])
        dialog.selectNameFilter(all_formats)
        if dialog.exec_():
            filename = dialog.selectedFiles()[0]
        else:
            return nothing
    ext = os.path.splitext(filename)[1][1:].lower()
    if ext in raw_exts:
        if RAWPY_AVAILABLE:
            with rawpy.imread(filename) as raw:
                image = cv.cvtColor(
                    raw.postprocess(
                        no_auto_bright=True,
                        use_camera_wb=True,
                    ),
                    cv.COLOR_RGB2BGR,
                )
        else:
            QMessageBox.warning(
                parent,
                parent.tr("Warning"),
                parent.tr("RAW image support not available. Please install rawpy or convert to JPEG/PNG first."),
            )
            return nothing
    elif ext == "gif":
        capture = cv.VideoCapture(filename)
        frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        if frames > 1:
            QMessageBox.warning(
                parent,
                parent.tr("Warning"),
                parent.tr("Animated GIF: importing first frame"),
            )
        result, image = capture.read()
        if not result:
            QMessageBox.critical(
                parent, parent.tr("Error"), parent.tr("Unable to decode GIF!")
            )
            return nothing
        if len(image.shape) == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image = cv.imread(filename, cv.IMREAD_COLOR)
    if image is None:
        QMessageBox.critical(
            parent, parent.tr("Error"), parent.tr("Unable to load image!")
        )
        return nothing
    if image.shape[2] > 3:
        QMessageBox.warning(
            parent, parent.tr("Warning"), parent.tr("Alpha channel discarded")
        )
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    settings.setValue("load_folder", QFileInfo(filename).absolutePath())
    return filename, os.path.basename(filename), image


def load_images(parent):
    """Load multiple images and return a list of (filename, basename, image) tuples."""
    settings = QSettings()
    mime_filters = [
        "image/jpeg",
        "image/png",
        "image/tiff",
        "image/gif",
        "image/bmp",
        "image/webp",
        "image/x-portable-pixmap",
        "image/x-portable-graymap",
        "image/x-portable-bitmap",
        "image/x-nikon-nef",
        "image/x-fuji-raf",
        "image/x-canon-cr2",
        "image/x-adobe-dng",
        "image/x-sony-arw",
        "image/x-kodak-dcr",
        "image/x-minolta-mrw",
        "image/x-pentax-pef",
        "image/x-canon-crw",
        "image/x-sony-sr2",
        "image/x-olympus-orf",
        "image/x-panasonic-raw",
    ]
    mime_db = QMimeDatabase()
    mime_patterns = [
        mime_db.mimeTypeForName(mime).globPatterns() for mime in mime_filters
    ]
    all_formats = f"Supported formats ({' '.join([item for sub in mime_patterns for item in sub])})"
    raw_exts = [p[-3:] for p in mime_patterns][-12:]
    dialog = QFileDialog(
        parent, parent.tr("Load images"), settings.value("load_folder")
    )
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.ExistingFiles)  # Allow multiple selection
    dialog.setViewMode(QFileDialog.Detail)
    dialog.setMimeTypeFilters(mime_filters)
    name_filters = dialog.nameFilters()
    dialog.setNameFilters(name_filters + [all_formats])
    dialog.selectNameFilter(all_formats)
    if dialog.exec_():
        filenames = dialog.selectedFiles()
    else:
        return []

    images = []
    for filename in filenames:
        ext = os.path.splitext(filename)[1][1:].lower()
        if ext in raw_exts:
            if RAWPY_AVAILABLE:
                with rawpy.imread(filename) as raw:
                    image = cv.cvtColor(
                        raw.postprocess(
                            no_auto_bright=True,
                            use_camera_wb=True,
                        ),
                        cv.COLOR_RGB2BGR,
                    )
            else:
                QMessageBox.warning(
                    parent,
                    parent.tr("Warning"),
                    parent.tr("RAW image support not available. Please install rawpy or convert to JPEG/PNG first."),
                )
                continue
        elif ext == "gif":
            capture = cv.VideoCapture(filename)
            frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
            if frames > 1:
                QMessageBox.warning(
                    parent,
                    parent.tr("Warning"),
                    parent.tr("Animated GIF: importing first frame"),
                )
            result, image = capture.read()
            if not result:
                QMessageBox.critical(
                    parent, parent.tr("Error"), parent.tr("Unable to decode GIF!")
                )
                continue
            if len(image.shape) == 2:
                image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        else:
            image = cv.imread(filename, cv.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(
                parent, parent.tr("Error"), parent.tr("Unable to load image!")
            )
            continue
        if image.shape[2] > 3:
            QMessageBox.warning(
                parent, parent.tr("Warning"), parent.tr("Alpha channel discarded")
            )
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        images.append((filename, os.path.basename(filename), image))
    if images:
        settings.setValue("load_folder", QFileInfo(filenames[0]).absolutePath())
    return images


def desaturate(image):
    return cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)


def norm_mat(matrix, to_bgr=False):
    norm = cv.normalize(matrix, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if not to_bgr:
        return norm
    return cv.cvtColor(norm, cv.COLOR_GRAY2BGR)


def exiftool_exe():
    if sys.platform.startswith("linux"):
        path = "pyexiftool/exiftool/linux/exiftool"
    elif sys.platform.startswith("win32"):
        path = "pyexiftool/exiftool/windows/exiftool(-k).exe"
    else:
        return None
    
    if os.path.exists(path):
        return path
    return None


def butter_exe():
    if sys.platform.startswith("linux"):
        return "butteraugli/linux/butteraugli"
    if sys.platform.startswith("win32"):
        return None
    return None


def ssimul_exe():
    if sys.platform.startswith("linux"):
        return "ssimulacra/linux/ssimulacra"
    if sys.platform.startswith("win32"):
        return None
    return None


class ParamSlider(QWidget):
    valueChanged = Signal(int)

    def __init__(
        self,
        interval,
        ticks=10,
        reset=0,
        suffix=None,
        label=None,
        bold=False,
        special=None,
        parent=None,
    ):
        super(ParamSlider, self).__init__(parent)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(interval[0], interval[1])
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval((interval[1] - interval[0] + 1) / ticks)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setValue(reset)
        self.slider.mouseDoubleClickEvent = self.doubleClicked

        self.spin = QSpinBox()
        self.spin.setRange(interval[0], interval[1])
        self.spin.setValue(reset)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(55)
        self.spin.setSpecialValueText(special)

        self.reset = reset
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.valueChanged)

        layout = QHBoxLayout()
        if label is not None:
            lab = QLabel(label)
            modify_font(lab, bold=bold)
            layout.addWidget(lab)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)
        self.setLayout(layout)

    def doubleClicked(self, _):
        self.reset_value()

    def reset_value(self):
        self.slider.setValue(self.reset)
        self.spin.setValue(self.reset)

    def value(self):
        return self.slider.value()

    def setValue(self, value):
        self.spin.setValue(value)
        self.slider.setValue(value)
        self.valueChanged.emit(value)

    def sync(self):
        self.spin.setValue(self.slider.value())
        self.slider.setValue(self.spin.value())
        self.valueChanged.emit(self.slider.value())


def _compute_bef(im, block_size=8):
    """Calculates Blocking Effect Factor (BEF) for a given grayscale/one channel image

    C. Yim and A. C. Bovik, "Quality Assessment of Deblocked Images," in IEEE Transactions on Image Processing,
        vol. 20, no. 1, pp. 88-98, Jan. 2011.

    :param im: input image (numpy ndarray)
    :param block_size: Size of the block over which DCT was performed during compression
    :return: float -- bef.
    """
    if len(im.shape) == 3:
        height, width, channels = im.shape
    elif len(im.shape) == 2:
        height, width = im.shape
        channels = 1
    else:
        raise ValueError("Not a 1-channel/3-channel grayscale image")

    if channels > 1:
        raise ValueError("Not for color images")

    h = np.array(range(0, width - 1))
    h_b = np.array(range(block_size - 1, width - 1, block_size))
    h_bc = np.array(list(set(h).symmetric_difference(h_b)))

    v = np.array(range(0, height - 1))
    v_b = np.array(range(block_size - 1, height - 1, block_size))
    v_bc = np.array(list(set(v).symmetric_difference(v_b)))

    d_b = 0
    d_bc = 0

    # h_b for loop
    for i in list(h_b):
        diff = im[:, i] - im[:, i+1]
        d_b += np.sum(np.square(diff))

    # h_bc for loop
    for i in list(h_bc):
        diff = im[:, i] - im[:, i+1]
        d_bc += np.sum(np.square(diff))

    # v_b for loop
    for j in list(v_b):
        diff = im[j, :] - im[j+1, :]
        d_b += np.sum(np.square(diff))

    # V_bc for loop
    for j in list(v_bc):
        diff = im[j, :] - im[j+1, :]
        d_bc += np.sum(np.square(diff))

    # N code
    n_hb = height * (width/block_size) - 1
    n_hbc = (height * (width - 1)) - n_hb
    n_vb = width * (height/block_size) - 1
    n_vbc = (width * (height - 1)) - n_vb

    # D code
    d_b /= (n_hb + n_vb)
    d_bc /= (n_hbc + n_vbc)

    # Log
    if d_b > d_bc:
        t = np.log2(block_size)/np.log2(min(height, width))
    else:
        t = 0

    # BEF
    bef = t*(d_b - d_bc)

    return bef


def psnrb(GT, P):
    """Calculates PSNR with Blocking Effect Factor for a given pair of images (PSNR-B)

    :param GT: first (original) input image in YCbCr format or Grayscale.
    :param P: second (corrected) input image in YCbCr format or Grayscale..
    :return: float -- psnr_b.
    """
    if len(GT.shape) == 3:
        GT = GT[:, :, 0]

    if len(P.shape) == 3:
        P = P[:, :, 0]

    imdff = np.double(GT) - np.double(P)

    mse = np.mean(np.square(imdff.flatten()))
    bef = _compute_bef(P)
    mse_b = mse + bef

    if np.amax(P) > 2:
        psnr_b = 10 * log10(255**2/mse_b)
    else:
        psnr_b = 10 * log10(1/mse_b)

    return psnr_b
