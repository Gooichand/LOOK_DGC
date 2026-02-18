# This code implements probability maps and fourier maps for detecting traces of resampling as explained in the paper:
# "Exposing Digital Forgeries by Detecting Traces of Re-sampling" by Hany Farid & Alin C. Popescu
# The book "Photo Forensics" by Hany Farid gives a more detailed explanation of the technique for those interested

from PySide6.QtWidgets import (
    QWidget,
    QDoubleSpinBox,
    QVBoxLayout,
    QSlider,
    QLabel,
    QCheckBox,
    QPushButton,
    QGridLayout,
    QSizePolicy,
    QScrollArea,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal
from tools import ToolWidget

# resampling necessary imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams["savefig.dpi"] = 200  # increase default save resolution
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from utility import modify_font


class ProbabilityWorker(QThread):
    finished_signal = Signal(object, object)  # probability_maps, image_copy

    def __init__(self, params, image_normalized, image_copy):
        super().__init__()
        self.params = params
        self.image_normalized = image_normalized
        self.image_copy = image_copy

    def run(self):
        probability_maps = []
        selected_points = self.params.get("selected_points", [])
        filter_5x5 = self.params.get("filter_5x5", False)
        
        if len(selected_points) == 0:
            if filter_5x5:
                processed_part = self.calculate_probability_map_5x5(
                    self.image_normalized
                )
            else:
                processed_part = self.calculate_probability_map_3x3(
                    self.image_normalized
                )

            probability_maps.append(processed_part)
            self.image_copy = processed_part
        else:
            for i in range(0, len(selected_points), 2):
                if i + 1 < len(selected_points):
                    (x1, y1) = selected_points[i]
                    (x2, y2) = selected_points[i + 1]
                    if filter_5x5:
                        processed_part = self.calculate_probability_map_5x5(
                            self.image_normalized[
                                min(y1, y2) : max(y1, y2) + 1,
                                min(x1, x2) : max(x1, x2) + 1,
                            ]
                        )
                        probability_maps.append(processed_part)
                        self.image_copy[
                            min(y1, y2) + 2 : max(y1, y2) - 1,
                            min(x1, x2) + 2 : max(x1, x2) - 1,
                        ] = processed_part
                    else:
                        processed_part = self.calculate_probability_map_3x3(
                            self.image_normalized[
                                min(y1, y2) : max(y1, y2) + 1,
                                min(x1, x2) : max(x1, x2) + 1,
                            ]
                        )
                        probability_maps.append(processed_part)
                        self.image_copy[
                            min(y1, y2) + 1 : max(y1, y2), min(x1, x2) + 1 : max(x1, x2)
                        ] = processed_part

        self.finished_signal.emit(probability_maps, self.image_copy)

    def calculate_probability_map_3x3(self, process_part):
        a = np.random.rand(8)
        a = a / a.sum()
        s = 0.005
        d = 0.1
        F, f = self.build_matrices_for_processing_3x3(process_part)
        c = 0
        w = np.zeros(F.shape[0])

        while c < 100:
            s2 = 0
            k = 0

            for y in range(1, process_part.shape[0] - 1):
                for x in range(1, process_part.shape[1] - 1):
                    r = self.compute_residual_3x3(a, process_part, x, y)
                    g = np.exp(-(r ** 2) / s)
                    w[k] = g / (g + d)
                    s2 = s2 + w[k] * r ** 2
                    k = k + 1

            s = s2 / w.sum()
            a2 = np.linalg.inv(F.T * w * w @ F) @ F.T * w * w @ f

            if np.linalg.norm(a - a2) < 0.01:
                break
            else:
                a = a2
                c = c + 1

        return w.reshape(process_part.shape[0] - 2, process_part.shape[1] - 2)

    def calculate_probability_map_5x5(self, process_part):
        a = np.random.rand(24)
        a = a / a.sum()
        s = 0.005
        d = 0.1
        F, f = self.build_matrices_for_processing_5x5(process_part)
        c = 0
        w = np.zeros(F.shape[0])

        while c < 100:
            s2 = 0
            k = 0

            for y in range(2, process_part.shape[0] - 2):
                for x in range(2, process_part.shape[1] - 2):
                    r = self.compute_residual_5x5(a, process_part, x, y)
                    g = np.exp(-(r ** 2) / s)
                    w[k] = g / (g + d)
                    s2 = s2 + w[k] * r ** 2
                    k = k + 1

            s = s2 / w.sum()
            a2 = np.linalg.inv(F.T * w * w @ F) @ F.T * w * w @ f

            if np.linalg.norm(a - a2) < 0.01:
                break
            else:
                a = a2
                c = c + 1

        return w.reshape(process_part.shape[0] - 4, process_part.shape[1] - 4)

    def build_matrices_for_processing_3x3(self, I):
        k = 0
        F = np.zeros(
            ((I.shape[0] - 2) * (I.shape[1] - 2), 8)
        )
        f = np.zeros(
            (I.shape[0] - 2) * (I.shape[1] - 2)
        )
        for y in range(1, I.shape[0] - 1):
            for x in range(1, I.shape[1] - 1):
                F[k, 0] = I[y - 1, x - 1]
                F[k, 1] = I[y - 1, x]
                F[k, 2] = I[y - 1, x + 1]
                F[k, 3] = I[y, x - 1]
                F[k, 4] = I[y, x + 1]
                F[k, 5] = I[y + 1, x - 1]
                F[k, 6] = I[y + 1, x]
                F[k, 7] = I[y + 1, x + 1]
                f[k] = I[y, x]
                k = k + 1
        return F, f

    def build_matrices_for_processing_5x5(self, I):
        k = 0
        F = np.zeros(
            ((I.shape[0] - 4) * (I.shape[1] - 4), 24)
        )
        f = np.zeros(
            (I.shape[0] - 4) * (I.shape[1] - 4)
        )
        for y in range(2, I.shape[0] - 2):
            for x in range(2, I.shape[1] - 2):
                F[k, 0] = I[y - 2, x - 2]
                F[k, 1] = I[y - 2, x - 1]
                F[k, 2] = I[y - 2, x]
                F[k, 3] = I[y - 2, x + 1]
                F[k, 4] = I[y - 2, x + 2]
                F[k, 5] = I[y - 1, x - 2]
                F[k, 6] = I[y - 1, x - 1]
                F[k, 7] = I[y - 1, x]
                F[k, 8] = I[y - 1, x + 1]
                F[k, 9] = I[y - 1, x + 2]
                F[k, 10] = I[y, x - 2]
                F[k, 11] = I[y, x - 1]
                F[k, 12] = I[y, x + 1]
                F[k, 13] = I[y, x + 2]
                F[k, 14] = I[y + 1, x - 2]
                F[k, 15] = I[y + 1, x - 1]
                F[k, 16] = I[y + 1, x]
                F[k, 17] = I[y + 1, x + 1]
                F[k, 18] = I[y + 1, x + 2]
                F[k, 19] = I[y + 2, x - 2]
                F[k, 20] = I[y + 2, x - 1]
                F[k, 21] = I[y + 2, x]
                F[k, 22] = I[y + 2, x + 1]
                F[k, 23] = I[y + 2, x + 2]
                f[k] = I[y, x]
                k = k + 1
        return F, f

    def compute_residual_3x3(self, a, I, x, y):
        r = I[y, x] - (
            a[0] * I[y - 1, x - 1]
            + a[1] * I[y - 1, x]
            + a[2] * I[y - 1, x + 1]
            + a[3] * I[y, x - 1]
            + a[4] * I[y, x + 1]
            + a[5] * I[y + 1, x - 1]
            + a[6] * I[y + 1, x]
            + a[7] * I[y + 1, x + 1]
        )
        return r

    def compute_residual_5x5(self, a, I, x, y):
        r = I[y, x] - (
            a[0] * I[y - 2, x - 2]
            + a[1] * I[y - 2, x - 1]
            + a[2] * I[y - 2, x]
            + a[3] * I[y - 2, x + 1]
            + a[4] * I[y - 2, x + 2]
            + a[5] * I[y - 1, x - 2]
            + a[6] * I[y - 1, x - 1]
            + a[7] * I[y - 1, x]
            + a[8] * I[y - 1, x + 1]
            + a[9] * I[y - 1, x + 2]
            + a[10] * I[y, x - 2]
            + a[11] * I[y, x - 1]
            + a[12] * I[y, x + 1]
            + a[13] * I[y, x + 2]
            + a[14] * I[y + 1, x - 2]
            + a[15] * I[y + 1, x - 1]
            + a[16] * I[y + 1, x]
            + a[17] * I[y + 1, x + 1]
            + a[18] * I[y + 1, x + 2]
            + a[19] * I[y + 2, x - 2]
            + a[20] * I[y + 2, x - 1]
            + a[21] * I[y + 2, x]
            + a[22] * I[y + 2, x + 1]
            + a[23] * I[y + 2, x + 2]
        )
        return r


class FourierWorker(QThread):
    map_ready_signal = Signal(int, object, object, object) # index, fourier_map, userfourplot_data, updated_fourier_maps_list
    finished_signal = Signal(list) # returns fourier_maps list

    def __init__(self, params, probability_maps, image_copy):
        super().__init__()
        self.params = params
        self.probability_maps = probability_maps
        self.image_copy = image_copy

    def run(self):
        fourier_maps = []
        selected_points_prob = self.params.get("selected_points_prob", [])
        selected_points_fourier = self.params.get("selected_points_fourier", [])
        
        i = 0
        counter_selected_prob_points = 0
        
        # Process probability maps
        for prob_map in self.probability_maps:
            if (
                len(self.probability_maps) == 1
                and len(selected_points_prob) == 0
            ):
                fourier_map = self.calculate_fourier_map(prob_map)
                fourier_maps.append(fourier_map)
                # Send None as userfourplot because we just show the prob_map itself in the original code
                self.map_ready_signal.emit(i, fourier_map, prob_map, fourier_maps)
                i = i + 1
            else:
                (x1, y1) = selected_points_prob[counter_selected_prob_points]
                (x2, y2) = selected_points_prob[counter_selected_prob_points + 1]
                
                fourier_map = self.calculate_fourier_map(prob_map)
                fourier_maps.append(fourier_map)
                
                userfourplot = self.image_copy.copy()
                self._draw_stripes(userfourplot, x1, y1, x2, y2)
                
                self.map_ready_signal.emit(i, fourier_map, userfourplot, fourier_maps)
                
                i = i + 1
                counter_selected_prob_points = counter_selected_prob_points + 2

        # Process Fourier windows
        for j in range(0, len(selected_points_fourier), 2):
            if j + 1 < len(selected_points_fourier):
                (x1, y1) = selected_points_fourier[j]
                (x2, y2) = selected_points_fourier[j + 1]

                processed_part = self.calculate_fourier_map(
                    self.image_copy[
                        min(y1, y2) : max(y1, y2) + 1, min(x1, x2) : max(x1, x2) + 1
                    ]
                )
                fourier_maps.append(processed_part)
                
                userfourplot = self.image_copy.copy()
                self._draw_stripes(userfourplot, x1, y1, x2, y2)
                
                self.map_ready_signal.emit(i, processed_part, userfourplot, fourier_maps)
                i = i + 1

        self.finished_signal.emit(fourier_maps)

    def _draw_stripes(self, img, x1, y1, x2, y2):
         img[
            min(y1, y2) : max(y1, y2) + 1,
            [
                x1,
                (x1 + 1),
                (x1 - 1),
                (x1 - 2),
                (x1 + 2),
                x2,
                (x2 + 1),
                (x2 - 1),
                (x2 - 2),
                (x2 + 2),
            ],
        ] = 0  # y-stripes
         img[
            [
                y1,
                (y1 - 1),
                (y1 + 1),
                (y1 - 2),
                (y1 + 2),
                y2,
                (y2 - 1),
                (y2 + 1),
                (y2 - 2),
                (y2 + 2),
            ],
            min(x1, x2) : max(x1, x2) + 1,
        ] = 0  # x-stripes

    def make_rotational_invariant_window(self, shape):
        rows, cols = shape
        W = np.zeros((rows, cols))
        center_x, center_y = rows // 2, cols // 2
        max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
        for i in range(rows):
            for j in range(cols):
                r = (
                    np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    / max_radius
                    * np.sqrt(2)
                )
                if r < 3 / 4:
                    W[i, j] = 1
                elif r <= np.sqrt(2):
                    W[i, j] = 0.5 + 0.5 * np.cos(
                        np.pi * (r - 3 / 4) / (np.sqrt(2) - 3 / 4)
                    )

        return W

    def make_high_pass_filter(self, shape):
        rows, cols = shape
        H = np.zeros((rows, cols))
        center_x, center_y = rows // 2, cols // 2
        max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
        for i in range(rows):
            for j in range(cols):
                r = (
                    np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                    / max_radius
                    * np.sqrt(2)
                )
                if r <= np.sqrt(2):
                    H[i, j] = 0.5 - 0.5 * np.cos(np.pi * r / np.sqrt(2))

        return H

    def calculate_fourier_map(self, process_prob_map):
        x, y = process_prob_map.shape
        size = min(x, y)
        half_size = size // 2
        center_x, center_y = x // 2, y // 2
        square_prob_map = process_prob_map[
            center_x - half_size : center_x + half_size,
            center_y - half_size : center_y + half_size,
        ]

        if self.params.get("hanning", False):
            hanning_window = np.hanning(square_prob_map.shape[0])[:, None] * np.hanning(
                square_prob_map.shape[1]
            )
            windowed_prob_map = square_prob_map * hanning_window
        elif self.params.get("rot_invariant", False):
            W = self.make_rotational_invariant_window(square_prob_map.shape)
            windowed_prob_map = square_prob_map * W
        else:
            return

        if self.params.get("upsample", False):
            upsampled = cv2.pyrUp(windowed_prob_map)
        else:
            upsampled = windowed_prob_map

        dft = np.fft.fft2(upsampled)
        fourier = np.fft.fftshift(dft)

        if self.params.get("center_four", False):
            height, width = fourier.shape
            center_x, center_y = width // 2, height // 2
            half_size = width // 4
            fourier_center = fourier[
                center_y - half_size : center_y + half_size,
                center_x - half_size : center_x + half_size,
            ]
        else:
            fourier_center = fourier

        if self.params.get("simple_highpass", False):
            rows, cols = fourier_center.shape
            center = (int(cols / 2), int(rows / 2))
            radius = int(0.1 * (min(rows, cols) / 2))
            if radius == 0:
                radius = 1

            Y, X = np.ogrid[:rows, :cols]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

            mask = dist_from_center <= radius
            filtered_spectrum = fourier_center.copy()
            filtered_spectrum[mask] = 0

        elif self.params.get("complex_highpass", False):
            H = self.make_high_pass_filter(fourier_center.shape)
            filtered_spectrum = fourier_center * H
        else:
            return

        magnitude_spectrum = np.abs(filtered_spectrum)
        scaled_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (
            magnitude_spectrum.max() - magnitude_spectrum.min()
        )

        gamma_corrected_spectrum = np.power(scaled_spectrum, self.params.get("gamma", 4))

        if self.params.get("rescale", False):
            rescaled_spectrum = gamma_corrected_spectrum * magnitude_spectrum.max()
        else:
            rescaled_spectrum = gamma_corrected_spectrum

        return rescaled_spectrum


class ResamplingWidget(ToolWidget):
    def __init__(self, filename, image, parent=None):
        super(ResamplingWidget, self).__init__(parent)

        self.filename = filename
        self.image_original = image
        self.imagegray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if self.imagegray is None:
            error_label = QLabel(self.tr("Unable to detect resampling on raw images!"))
            modify_font(error_label, bold=True)
            error_label.setStyleSheet("color: #FF0000")
            error_label.setAlignment(Qt.AlignCenter)
            main_layout = QVBoxLayout()
            main_layout.addWidget(error_label)
            self.setLayout(main_layout)
            return
        self.imagegray = self.imagegray - self.imagegray.min()
        self.imagegray = self.imagegray / self.imagegray.max()

        self.imagegray_nomalized_copy = self.imagegray.copy()
        self.imagegray_copy_for_probabilitymaps = self.imagegray.copy()
        self.userfourplot = self.imagegray.copy()

        self.selected_points_probability = []
        self.selected_points_fourier = []
        self.probability_maps = []
        self.fourier_maps = []
        self.original_sizes_widgets = {}
        
        self.prob_worker = None
        self.four_worker = None

        self.calculate_probability_button = QPushButton(
            self.tr("Calculate probability")
        )
        self.calculate_fourier_button = QPushButton(self.tr("Calculate fourier"))

        self.filter_3x3_Check = QCheckBox(self.tr("3x3 probability filter"))
        self.filter_3x3_Check.setChecked(True)
        self.filter_5x5_Check = QCheckBox(self.tr("5x5 probability filter"))
        self.filter_5x5_Check.setChecked(False)

        self.filter_3x3_Check.clicked.connect(
            lambda: self.filter_5x5_Check.setChecked(False)
        )
        self.filter_5x5_Check.clicked.connect(
            lambda: self.filter_3x3_Check.setChecked(False)
        )

        self.probability_check = QCheckBox(self.tr("Probability Windows"))
        self.probability_check.setChecked(True)

        self.fourier_check = QCheckBox(self.tr("Fourier Windows"))
        self.fourier_check.setChecked(False)

        top_layout = QGridLayout()
        top_layout.addWidget(
            QLabel(
                self.tr(
                    "A resampling analysis is computationally expensive and for a\n"
                    + "high resoluion image (1280x720 pixels), it can take a few minutes to process.\n"
                    + "It is more efficient and more effective to calculate the probability map for only the suspected area.\n"
                    + "This is because a small resampled area is more readily "
                    + "detected when the larger image is not considered by the algorithm."
                )
            ),
            0,
            0,
        )

        top_layout.addWidget(
            QLabel(
                self.tr(
                    "A 5x5 probability filter might be able to uncover more complex interpolation algorithms, but the computing time for the probability maps increases significantly.\n"
                    + "Left mouse button: choose location to construct area of interest\n"
                    + "Right mouse button: delete last location."
                )
            ),
            1,
            0,
        )

        top_layout.addWidget(
            QLabel(
                self.tr(
                    "If no area of interest is chosen for probability, the probability map for the entire image will be calculated.\n"
                    + "Fourier maps will automatically be calculated for each area of interest by the probability map + for smaller sub windows applied when 'Fourier Windows' is checked.\n"
                    "Please make sure areas do not overlap for probability maps!"
                )
            ),
            2,
            0,
        )

        top_layout.addWidget(self.probability_check, 0, 1)
        top_layout.addWidget(self.fourier_check, 0, 2)

        checkbox_layout = QVBoxLayout()
        checkbox_layout.addWidget(self.filter_3x3_Check)
        checkbox_layout.addWidget(self.filter_5x5_Check)
        top_layout.addLayout(checkbox_layout, 1, 1)

        top_layout.addWidget(self.calculate_probability_button, 2, 1)
        top_layout.addWidget(self.calculate_fourier_button, 2, 2)

        four_parameters_layout = QGridLayout()
        self.hanning_check = QCheckBox(self.tr("Hanning"))
        self.hanning_check.setChecked(True)
        self.rotationally_invariant_window_check = QCheckBox(self.tr("R. I. W."))
        self.rotationally_invariant_window_check.setChecked(False)
        self.hanning_check.clicked.connect(
            lambda: self.rotationally_invariant_window_check.setChecked(False)
        )
        self.rotationally_invariant_window_check.clicked.connect(
            lambda: self.hanning_check.setChecked(False)
        )

        self.upsample_check = QCheckBox(self.tr("Upsample"))
        self.upsample_check.setChecked(True)
        self.center_four_check = QCheckBox(self.tr("Take center of Fourier"))
        self.center_four_check.setChecked(False)

        self.simple_highpass_check = QCheckBox(self.tr("Highpass 1"))
        self.simple_highpass_check.setChecked(True)
        self.complex_highpass_check = QCheckBox(self.tr("Highpass 2"))
        self.complex_highpass_check.setChecked(False)
        self.simple_highpass_check.clicked.connect(
            lambda: self.complex_highpass_check.setChecked(False)
        )
        self.complex_highpass_check.clicked.connect(
            lambda: self.simple_highpass_check.setChecked(False)
        )

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0, 5)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(4)

        self.rescale_check = QCheckBox(self.tr("Rescale Spectrum"))
        self.rescale_check.setChecked(True)

        four_parameters_layout.addWidget(self.hanning_check, 0, 0)
        four_parameters_layout.addWidget(self.rotationally_invariant_window_check, 1, 0)
        four_parameters_layout.addWidget(self.upsample_check, 1, 1)
        four_parameters_layout.addWidget(self.center_four_check, 1, 2)
        four_parameters_layout.addWidget(self.simple_highpass_check, 0, 3)
        four_parameters_layout.addWidget(self.complex_highpass_check, 1, 3)
        four_parameters_layout.addWidget(QLabel(self.tr("Gamma correction")), 0, 4)
        four_parameters_layout.addWidget(self.gamma_spin, 1, 4)
        four_parameters_layout.addWidget(self.rescale_check, 1, 5)

        figure_prob = Figure()
        self.canvas = FigureCanvas(figure_prob)
        self.axes = self.canvas.figure.subplots()
        self.probability_image_canvas_object = self.axes.imshow(
            self.imagegray, cmap="gray"
        )
        self.canvas.mpl_connect("button_press_event", self.click_on_canvas)
        self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.axes.figure.canvas.draw()

        self.toolbar_prob = NavigationToolbar(self.canvas, self)

        self.figure_four = plt.figure(figsize=(10, 15))
        self.canvas_fourier_maps = FigureCanvas(self.figure_four)
        self.canvas_fourier_maps.mpl_connect("button_press_event", self.click_on_canvas)
        self.canvas_fourier_maps.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.toolbar_four = NavigationToolbar(self.canvas_fourier_maps, self)

        self.calculate_probability_button.clicked.connect(
            self.calculate_probability_maps
        )
        self.calculate_fourier_button.clicked.connect(self.calculate_fourier_maps)
        self.probability_check.clicked.connect(
            lambda: self.fourier_check.setChecked(False)
        )
        self.fourier_check.clicked.connect(
            lambda: self.probability_check.setChecked(False)
        )

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        self.four_parameters_label = QLabel(
            "Fourier map options: (Make sure to have Hanning or R. I. W. checked, same for Highpass 1 or 2.)"
        )
        main_layout.addWidget(self.four_parameters_label)
        main_layout.addLayout(four_parameters_layout)

        self.zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(100)
        self.zoom_slider.setMaximum(200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(5)
        self.zoom_slider.valueChanged.connect(self.slider_zoom)

        main_layout.addWidget(self.zoom_label)
        main_layout.addWidget(self.zoom_slider)

        self.scroll_area = QScrollArea(self)
        self.scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)

        scroll_layout.addWidget(self.toolbar_prob)
        scroll_layout.addWidget(self.canvas)
        self.original_sizes_widgets[self.canvas] = self.canvas.sizeHint()

        scroll_layout.addWidget(self.toolbar_four)
        scroll_layout.addWidget(self.canvas_fourier_maps)
        self.original_sizes_widgets[
            self.canvas_fourier_maps
        ] = self.canvas_fourier_maps.sizeHint()

        main_layout.addWidget(self.scroll_area)

        self.setLayout(main_layout)

    def slider_zoom(self):
        zoom_factor = self.zoom_slider.value() / 100
        for child_widget in self.scroll_widget.findChildren(QWidget):
            if child_widget in self.original_sizes_widgets:
                original_size = self.original_sizes_widgets[child_widget]
                new_width = original_size.width() * zoom_factor
                new_height = original_size.height() * zoom_factor
                child_widget.setFixedSize(new_width, new_height)

    def calculate_probability_maps(self):
        self.probability_maps = []
        # Reset image copy for new calculation (based on normalized copy)
        self.imagegray_copy_for_probabilitymaps = self.imagegray_nomalized_copy.copy()
        
        self.calculate_probability_button.setEnabled(False)
        self.calculate_probability_button.setText(self.tr("Processing..."))
        
        params = {
            "selected_points": self.selected_points_probability,
            "filter_5x5": self.filter_5x5_Check.isChecked()
        }
        
        # Pass a copy of the image data to avoid threading issues
        self.prob_worker = ProbabilityWorker(
            params, 
            self.imagegray_nomalized_copy.copy(),
            self.imagegray_copy_for_probabilitymaps.copy()
        )
        self.prob_worker.finished_signal.connect(self.on_probability_finished)
        self.prob_worker.start()

    def on_probability_finished(self, probability_maps, image_copy):
        self.probability_maps = probability_maps
        self.imagegray_copy_for_probabilitymaps = image_copy
        
        self.probability_image_canvas_object.set_data(
            self.imagegray_copy_for_probabilitymaps
        )
        self.axes.figure.canvas.draw()
        
        self.calculate_probability_button.setEnabled(True)
        self.calculate_probability_button.setText(self.tr("Calculate probability"))

    def calculate_fourier_maps(self):
        self.fourier_maps = []
        self.calculate_fourier_button.setEnabled(False)
        self.calculate_fourier_button.setText(self.tr("Processing..."))

        num_plots = int(len(self.selected_points_fourier) / 2) + len(
            self.probability_maps
        )
        self.canvas_fourier_maps.figure.clf()
        if num_plots > 1:
            self.axes_fourier_maps = self.canvas_fourier_maps.figure.subplots(
                num_plots, 2
            )
        else:
            self.axes_fourier_maps = np.array(
                [
                    [
                        self.canvas_fourier_maps.figure.add_subplot(1, 2, 1),
                        self.canvas_fourier_maps.figure.add_subplot(1, 2, 2),
                    ]
                ]
            )
            
        params = {
            "selected_points_prob": self.selected_points_probability,
            "selected_points_fourier": self.selected_points_fourier,
            "hanning": self.hanning_check.isChecked(),
            "rot_invariant": self.rotationally_invariant_window_check.isChecked(),
            "upsample": self.upsample_check.isChecked(),
            "center_four": self.center_four_check.isChecked(),
            "simple_highpass": self.simple_highpass_check.isChecked(),
            "complex_highpass": self.complex_highpass_check.isChecked(),
            "gamma": self.gamma_spin.value(),
            "rescale": self.rescale_check.isChecked()
        }
        
        self.four_worker = FourierWorker(
            params,
            self.probability_maps,
            self.imagegray_copy_for_probabilitymaps.copy()
        )
        self.four_worker.map_ready_signal.connect(self.on_fourier_map_ready)
        self.four_worker.finished_signal.connect(self.on_fourier_finished)
        self.four_worker.start()

    def on_fourier_map_ready(self, index, fourier_map, userfourplot, fourier_maps_list):
        self.fourier_maps = fourier_maps_list # Sync list
        
        self.axes_fourier_maps[index, 0].imshow(
            userfourplot, cmap="gray", vmin=0, vmax=1
        )
        self.axes_fourier_maps[index, 0].axis("off")
        
        self.axes_fourier_maps[index, 1].imshow(
            fourier_map, cmap="gray", vmin=0, vmax=1
        )
        self.axes_fourier_maps[index, 1].axis("off")
        
        # Adjust layout only once or periodically? Doing it here might be expensive but ensures correct layout
        self.figure_four.subplots_adjust(
            left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05
        )
        self.canvas_fourier_maps.figure.canvas.draw()

    def on_fourier_finished(self, fourier_maps):
        self.fourier_maps = fourier_maps
        self.calculate_fourier_button.setEnabled(True)
        self.calculate_fourier_button.setText(self.tr("Calculate fourier"))

    def click_on_canvas(self, event):
        if self.toolbar_four.mode == "" and self.toolbar_prob.mode == "":
            if event.inaxes:
                if self.probability_check.isChecked():
                    if event.button == MouseButton.LEFT:
                        x, y = int(event.xdata), int(event.ydata)
                        self.selected_points_probability.append((x, y))
                        print(f"Selected point: ({x}, {y})")
                        self.imagegray = self.imagegray_nomalized_copy.copy()
                        self.draw_selection(self.selected_points_probability)
                        self.probability_image_canvas_object.set_data(self.imagegray)
                        self.axes.figure.canvas.draw()
                    elif (
                        event.button == MouseButton.RIGHT
                        and self.selected_points_probability
                    ):
                        print(self.selected_points_probability)
                        self.selected_points_probability.pop()
                        print(self.selected_points_probability)
                        self.imagegray = self.imagegray_nomalized_copy.copy()
                        self.draw_selection(self.selected_points_probability)
                        self.probability_image_canvas_object.set_data(self.imagegray)
                        self.axes.figure.canvas.draw()
                elif self.fourier_check.isChecked():
                    if event.button == MouseButton.LEFT:
                        x, y = int(event.xdata), int(event.ydata)
                        self.selected_points_fourier.append((x, y))
                        self.userfourplot = (
                            self.imagegray_copy_for_probabilitymaps.copy()
                        )
                        self.draw_selection(self.selected_points_fourier)
                        self.probability_image_canvas_object.set_data(self.userfourplot)
                        self.axes.figure.canvas.draw()
                    elif (
                        event.button == MouseButton.RIGHT
                        and self.selected_points_fourier
                    ):
                        self.selected_points_fourier.pop()
                        self.userfourplot = (
                            self.imagegray_copy_for_probabilitymaps.copy()
                        )
                        self.draw_selection(self.selected_points_fourier)
                        self.probability_image_canvas_object.set_data(self.userfourplot)
                        self.axes.figure.canvas.draw()

    def draw_selection(self, selected_points):
        i = 1
        if self.probability_check.isChecked():
            for point in selected_points:
                cv2.circle(self.imagegray, point, 3, (1,), -1)
                if (i % 2) == 0:
                    self.draw_rectangle(selected_points[i - 2], selected_points[i - 1])

                i = i + 1
        elif self.fourier_check.isChecked():
            for point in selected_points:
                cv2.circle(self.userfourplot, point, 3, (1,), -1)
                if (i % 2) == 0:
                    self.draw_rectangle(selected_points[i - 2], selected_points[i - 1])

                i = i + 1

    def draw_rectangle(self, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        if self.probability_check.isChecked():
            self.imagegray[
                min(y1, y2) : max(y1, y2) + 1,
                [x1, (x1 + 1), (x1 - 1), x2, (x2 + 1), (x2 - 1)],
            ] = 0  # y-stripes
            self.imagegray[
                [y1, (y1 - 1), (y1 + 1), y2, (y2 - 1), (y2 + 1)],
                min(x1, x2) : max(x1, x2) + 1,
            ] = 0  # x-stripes
        elif self.fourier_check.isChecked():
            self.userfourplot[
                min(y1, y2) : max(y1, y2) + 1,
                [x1, (x1 + 1), (x1 - 1), x2, (x2 + 1), (x2 - 1)],
            ] = 0  # y-stripes
            self.userfourplot[
                [y1, (y1 - 1), (y1 + 1), y2, (y2 - 1), (y2 + 1)],
                min(x1, x2) : max(x1, x2) + 1,
            ] = 0  # x-stripes
