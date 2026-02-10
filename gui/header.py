import os
import sys
from subprocess import run, PIPE

from PySide6.QtCore import QUrl, QTemporaryDir, QThread, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QLabel

from tools import ToolWidget
from utility import exiftool_exe

class HeaderWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        
    def run(self):
        try:
            exiftool_path = exiftool_exe()
            if not exiftool_path or not os.path.exists(exiftool_path):
                self.error.emit("ExifTool not found - please install ExifTool")
                return
                
            p = run([exiftool_path, "-htmldump0", self.filename], 
                   stdout=PIPE, stderr=PIPE, timeout=30)
            
            if p.returncode != 0:
                self.error.emit("Unable to analyze file header structure")
                return
                
            html_content = p.stdout.decode("utf-8")
            if len(html_content) < 100:
                self.error.emit("No header structure data available")
                return
                
            self.finished.emit(html_content)
        except Exception as e:
            self.error.emit(str(e))


class HeaderWidget(ToolWidget):
    def __init__(self, filename, parent=None):
        super(HeaderWidget, self).__init__(parent)
        
        self.loading_label = QLabel("Loading header structure...")
        layout = QVBoxLayout()
        layout.addWidget(self.loading_label)
        self.setLayout(layout)
        self.setMinimumWidth(900)
        
        self.temp_dir = QTemporaryDir()
        self.worker = HeaderWorker(filename)
        self.worker.finished.connect(self.on_data_ready)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        
    def on_data_ready(self, html_content):
        if self.temp_dir.isValid():
            temp_file = os.path.join(self.temp_dir.path(), "structure.html")
            with open(temp_file, "w") as file:
                file.write(html_content)

            web_view = QWebEngineView()
            if sys.platform.startswith("win32"):
                temp_file = self.temp_dir.path() + "//" + "structure.html"
                web_view.load(QUrl(temp_file))
            else:
                web_view.load(QUrl("file://" + temp_file))

            # Clear layout and add web view
            layout = self.layout()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            layout.addWidget(web_view)
        
    def on_error(self, error_msg):
        error_label = QLabel(f"Error loading header: {error_msg}")
        error_label.setStyleSheet("color: red;")

        layout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        layout.addWidget(error_label)

    def get_report_data(self):
        """Return data for PDF report generation"""
        text = "File Header Analysis:\n"
        text += "Analyzes the file header structure using ExifTool.\n"
        text += "Provides detailed metadata and header information in HTML format."

        # Since header analysis is HTML-based, we return text summary
        # The actual HTML content is displayed in the widget but not easily extractable for PDF
        return {'text': text}
