from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class MultipleCompressionWidget(QWidget):
    info_message = Signal(str)

    def __init__(self, filename, image, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.image = image

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Multiple Compression analysis"))

        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        # Separate buttons for graph and heatmap
        graph_button = QPushButton("Show Graph")
        graph_button.clicked.connect(self.run_graph)
        layout.addWidget(graph_button)

        heatmap_button = QPushButton("Show Heatmap")
        heatmap_button.clicked.connect(self.run_heatmap)
        layout.addWidget(heatmap_button)

        # Graph canvas
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Heatmap placeholder
        self.heatmap_label = QLabel()
        layout.addWidget(self.heatmap_label)

        self.setLayout(layout)
        self.info_message.emit("Detects multiple JPEG compressions")

    def run_graph(self):
        qualities = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
        diffs = []
        for q in qualities:
            _, encimg = cv2.imencode('.jpg', self.image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            decimg = cv2.imdecode(encimg, 1)
            diff = np.mean(cv2.absdiff(self.image, decimg))
            diffs.append((q, diff))

        msg = f"Recompression diffs: {diffs}"
        self.info_message.emit(msg)
        self.result_label.setText(msg)

        # Update graph
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        qualities, values = zip(*diffs)
        ax.plot(qualities, values, marker='o')
        ax.set_xlabel("JPEG Quality")
        ax.set_ylabel("Difference")
        ax.set_title("Multiple Compression Analysis")
        self.canvas.draw()

    def run_heatmap(self):
        _, encimg = cv2.imencode('.jpg', self.image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        decimg = cv2.imdecode(encimg, 1)
        diff_img = cv2.absdiff(self.image, decimg)
        heatmap = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
        h, w, ch = heatmap.shape
        qimg = QImage(heatmap.data, w, h, ch * w, QImage.Format_BGR888)
        self.heatmap_label.setPixmap(QPixmap.fromImage(qimg))
