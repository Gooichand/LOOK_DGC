from PySide6.QtWidgets import QVBoxLayout

from tools import ToolWidget
from viewer import ImageViewer


class OriginalWidget(ToolWidget):
    def __init__(self, image, parent=None):
        super(OriginalWidget, self).__init__(parent)
        self.image = image
        viewer = ImageViewer(image, None, None)
        layout = QVBoxLayout()
        layout.addWidget(viewer)
        self.setLayout(layout)

    def get_report_data(self):
        """Return data for PDF report generation"""
        text = "Original Image:\n"
        text += "Displays the original image without any processing.\n"
        text += "This serves as the baseline for comparison with other analysis tools."

        return {
            'text': text,
            'image': self.image
        }
