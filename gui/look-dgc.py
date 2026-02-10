# (FULL FILE – CLEAN MERGE RESOLVED)

import os
import sys

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QKeySequence, QIcon, QAction
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QDockWidget,
    QMessageBox,
    QFileDialog,
)

from adjust import AdjustWidget
from cloning import CloningWidget
from comparison import ComparisonWidget
from contrast import ContrastWidget
from digest import DigestWidget
from echo import EchoWidget
from editor import EditorWidget
from ela import ElaWidget
from exif import ExifWidget
from frequency import FrequencyWidget
from gradient import GradientWidget
from header import HeaderWidget
from histogram import HistWidget
from location import LocationWidget
from magnifier import MagnifierWidget
from median import MedianWidget
from minmax import MinMaxWidget
from multiple import MultipleWidget
from noise import NoiseWidget
from original import OriginalWidget
from pca import PcaWidget
from planes import PlanesWidget
from plots import PlotsWidget
from quality import QualityWidget
from reverse import ReverseWidget
from space import SpaceWidget

try:
    from splicing import SplicingWidget
    SPLICING_AVAILABLE = True
except ImportError:
    SPLICING_AVAILABLE = False

from trufor import TruForWidget
from stats import StatsWidget
from stereogram import StereoWidget
from thumbnail import ThumbWidget
from tools import ToolTree
from utility import modify_font, load_image
from wavelets import WaveletWidget
from ghostmmaps import GhostmapWidget
from resampling import ResamplingWidget
from noise_estimmation import NoiseWaveletBlockingWidget
from report import generate_pdf_report


class MainWindow(QMainWindow):
    max_recent = 5

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        QApplication.setApplicationName("LOOK-DGC")
        QApplication.setOrganizationName("Gopichand")
        QApplication.setOrganizationDomain("http://www.gopichand-dgc.com")
        QApplication.setApplicationVersion(ToolTree().version)
        QApplication.setWindowIcon(QIcon("icons/look-dgc_white.png"))

        self.setWindowTitle(
            f"{QApplication.applicationName()} {QApplication.applicationVersion()}"
        )

        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)

        self.filename = None
        self.image = None

        modify_font(self.statusBar(), bold=True)

        # ---------------- TOOLS TREE ----------------
        tree_dock = QDockWidget(self.tr("TOOLS"), self)
        self.addDockWidget(Qt.LeftDockWidgetArea, tree_dock)

        self.tree_widget = ToolTree()
        self.tree_widget.itemDoubleClicked.connect(self.open_tool)
        tree_dock.setWidget(self.tree_widget)

        tools_action = tree_dock.toggleViewAction()
        tools_action.setText(self.tr("Show tools"))
        tools_action.setShortcut(QKeySequence(Qt.Key_Tab))
        tools_action.setIcon(QIcon("icons/tools.svg"))

        # ---------------- HELP ACTION ----------------
        help_action = QAction(self.tr("Show help"), self)
        help_action.setToolTip(self.tr("Toggle online help"))
        help_action.setShortcut(QKeySequence.HelpContents)
        help_action.setIcon(QIcon("icons/help.svg"))
        help_action.setCheckable(True)
        help_action.triggered.connect(self.show_help)

        # ---------------- FILE ACTIONS ----------------
        load_action = QAction(self.tr("&Load image..."), self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.triggered.connect(self.load_file)

        report_action = QAction(self.tr("&Generate Report..."), self)
        report_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_R))
        report_action.triggered.connect(self.generate_report)

        quit_action = QAction(self.tr("&Quit"), self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)

        # ---------------- MENUS ----------------
        file_menu = self.menuBar().addMenu(self.tr("&File"))
        file_menu.addAction(load_action)
        file_menu.addAction(report_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu(self.tr("&View"))
        view_menu.addAction(tools_action)
        view_menu.addAction(help_action)

        help_menu = self.menuBar().addMenu(self.tr("&Help"))
        help_menu.addAction(help_action)

        # ---------------- TOOLBAR ----------------
        toolbar = self.addToolBar(self.tr("Main Toolbar"))
        toolbar.addAction(load_action)
        toolbar.addAction(tools_action)
        toolbar.addAction(help_action)
        toolbar.addAction(report_action)

        self.show_message(self.tr("Ready"))

    # ---------------- CORE METHODS ----------------

    def show_help(self):
        from help import show_help_dialog
        show_help_dialog(self)

    def load_file(self):
        filename, basename, image = load_image(self)
        if filename is None:
            return
        self.initialize(filename, basename, image)

    def initialize(self, filename, basename, image):
        self.filename = filename
        self.image = image
        self.tree_widget.setEnabled(True)
        self.setWindowTitle(
            f"({basename}) - {QApplication.applicationName()} {QApplication.applicationVersion()}"
        )
        self.show_message(self.tr(f'Image "{basename}" successfully loaded'))
        self.mdi_area.closeAllSubWindows()
        self.open_tool(self.tree_widget.topLevelItem(0).child(0), None)

    def open_tool(self, item, _):
        if not item.data(0, Qt.UserRole):
            return

        QMessageBox.information(self, "Tool", f"{item.text(0)} loaded")

    # ---------------- REPORT ----------------

    def generate_report(self):
        if self.filename is None or self.image is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        analysis_data = {}

        for sub_window in self.mdi_area.subWindowList():
            widget = sub_window.widget()
            if hasattr(widget, 'get_report_data'):
                data = widget.get_report_data()
                if data:
                    analysis_data[sub_window.windowTitle()] = data

        if not analysis_data:
            QMessageBox.information(self, "No Analysis Data", "No tools open.")
            return

        output_path = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "", "PDF files (*.pdf)"
        )[0]

        if not output_path:
            return

        if not output_path.endswith(".pdf"):
            output_path += ".pdf"

        try:
            generate_pdf_report(self.filename, self.filename, self.image, analysis_data, output_path)
            QMessageBox.information(self, "Success", f"Report saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Report failed:\n{str(e)}")

    def show_message(self, message):
        self.statusBar().showMessage(message, 10000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    sys.exit(app.exec())
