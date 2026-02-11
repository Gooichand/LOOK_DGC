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
from batch import BatchAnalysisWidget
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

        tree_dock = QDockWidget(self.tr("TOOLS"), self)
        tree_dock.setObjectName("tree_dock")
        tree_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, tree_dock)
        self.tree_widget = ToolTree()
        self.tree_widget.setObjectName("tree_widget")
        self.tree_widget.itemDoubleClicked.connect(self.open_tool)
        tree_dock.setWidget(self.tree_widget)

        tools_action = tree_dock.toggleViewAction()
        tools_action.setText(self.tr("Show tools"))
        tools_action.setToolTip(self.tr("Toggle toolset visibility"))
        tools_action.setShortcut(QKeySequence(Qt.Key_Tab))
        tools_action.setObjectName("tools_action")
        tools_action.setIcon(QIcon("icons/tools.svg"))

        help_action = QAction(self.tr("Show help "), self)
        help_action.setToolTip(self.tr("Toggle online help (F1)"))
        help_action.setShortcut(QKeySequence(Qt.Key_F1))
        help_action.setShortcutContext(Qt.ApplicationShortcut)
        help_action.setObjectName("help_action")
        help_action.setIcon(QIcon("icons/help.svg"))
        help_action.triggered.connect(self.show_help)
        self.addAction(help_action)

        load_action = QAction(self.tr("&Load image..."), self)
        load_action.setToolTip(self.tr("Load an image to analyze"))
        load_action.setShortcut(QKeySequence.Open)
        load_action.triggered.connect(self.load_file)
        load_action.setObjectName("load_action")
        load_action.setIcon(QIcon("icons/load.svg"))

        report_action = QAction(self.tr("&Generate Report..."), self)
        report_action.setToolTip(self.tr("Generate PDF report of analysis results"))
        report_action.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_R))
        report_action.triggered.connect(self.generate_report)
        report_action.setObjectName("report_action")
        report_action.setIcon(QIcon("icons/export.svg"))

        quit_action = QAction(self.tr("&Quit"), self)
        quit_action.setToolTip(self.tr("Exit from LOOK-DGC"))
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        quit_action.setObjectName("quit_action")
        quit_action.setIcon(QIcon("icons/quit.svg"))

        tabbed_action = QAction(self.tr("&Tabbed"), self)
        tabbed_action.setToolTip(self.tr("Toggle tabbed view for window area"))
        tabbed_action.setShortcut(QKeySequence(Qt.Key_F10))
        tabbed_action.setCheckable(True)
        tabbed_action.triggered.connect(self.toggle_view)
        tabbed_action.setObjectName("tabbed_action")
        tabbed_action.setIcon(QIcon("icons/tabbed.svg"))

        prev_action = QAction(self.tr("&Previous"), self)
        prev_action.setToolTip(self.tr("Select the previous tool window"))
        prev_action.setShortcut(QKeySequence.PreviousChild)
        prev_action.triggered.connect(self.mdi_area.activatePreviousSubWindow)
        prev_action.setObjectName("prev_action")
        prev_action.setIcon(QIcon("icons/previous.svg"))

        next_action = QAction(self.tr("&Next"), self)
        next_action.setToolTip(self.tr("Select the next tool window"))
        next_action.setShortcut(QKeySequence.NextChild)
        next_action.triggered.connect(self.mdi_area.activateNextSubWindow)
        next_action.setObjectName("next_action")
        next_action.setIcon(QIcon("icons/next.svg"))

        tile_action = QAction(self.tr("&Tile\tF9"), self)
        tile_action.setToolTip(self.tr("Arrange windows into non-overlapping views (F9)"))
        tile_action.setObjectName("tile_action")
        tile_action.setIcon(QIcon("icons/tile.svg"))

        cascade_action = QAction(self.tr("&Cascade\tF12"), self)
        cascade_action.setToolTip(self.tr("Arrange windows into overlapping views (F12)"))
        cascade_action.setObjectName("cascade_action")
        cascade_action.setIcon(QIcon("icons/cascade.svg"))

        close_action = QAction(self.tr("Close &All"), self)
        close_action.setToolTip(self.tr("Close all open tool windows"))
        close_action.setShortcut(QKeySequence(Qt.CTRL | Qt.SHIFT | Qt.Key_W))
        close_action.triggered.connect(self.mdi_area.closeAllSubWindows)
        close_action.setObjectName("close_action")
        close_action.setIcon(QIcon("icons/close.svg"))

        self.full_action = QAction(self.tr("Full screen\tF5"), self)
        self.full_action.setToolTip(self.tr("Switch to full screen mode (F5)"))
        self.full_action.setObjectName("full_action")
        self.full_action.setIcon(QIcon("icons/full.svg"))
        self.full_action.triggered.connect(self.change_view)

        self.normal_action = QAction(self.tr("Normal view\tF5"), self)
        self.normal_action.setToolTip(self.tr("Back to normal view mode (F5)"))
        self.normal_action.setObjectName("normal_action")
        self.normal_action.setIcon(QIcon("icons/normal.svg"))
        self.normal_action.triggered.connect(self.change_view)

        about_action = QAction(self.tr("&About..."), self)
        about_action.setToolTip(self.tr("Information about this program"))
        about_action.triggered.connect(self.show_about)
        about_action.setObjectName("about_action")
        about_action.setIcon(QIcon("icons/look-dgc_alpha.png"))

        about_qt_action = QAction(self.tr("About &Qt"), self)
        about_qt_action.setToolTip(self.tr("Information about the Qt Framework"))
        about_qt_action.triggered.connect(QApplication.aboutQt)
        about_qt_action.setIcon(QIcon("icons/Qt.svg"))

        file_menu = self.menuBar().addMenu(self.tr("&File"))
        file_menu.addAction(load_action)
        file_menu.addSeparator()
        self.recent_actions = [None] * self.max_recent
        for i in range(len(self.recent_actions)):
            self.recent_actions[i] = QAction(self)
            self.recent_actions[i].setVisible(False)
            self.recent_actions[i].triggered.connect(self.open_recent)
            file_menu.addAction(self.recent_actions[i])
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu(self.tr("&View"))
        view_menu.addAction(tools_action)
        view_menu.addAction(help_action)
        view_menu.addSeparator()
        view_menu.addAction(self.full_action)
        view_menu.addAction(self.normal_action)

        window_menu = self.menuBar().addMenu(self.tr("&Window"))
        window_menu.addAction(prev_action)
        window_menu.addAction(next_action)
        window_menu.addSeparator()
        window_menu.addAction(tile_action)
        window_menu.addAction(cascade_action)
        window_menu.addAction(tabbed_action)
        window_menu.addSeparator()
        window_menu.addAction(close_action)

        help_menu = self.menuBar().addMenu(self.tr("&Help"))
        help_menu.addAction(help_action)
        help_menu.addSeparator()
        help_menu.addAction(about_action)
        help_menu.addAction(about_qt_action)

        main_toolbar = self.addToolBar(self.tr("&Toolbar"))
        main_toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        main_toolbar.addAction(load_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(tools_action)
        main_toolbar.addAction(help_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(prev_action)
        main_toolbar.addAction(next_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(tile_action)
        main_toolbar.addAction(cascade_action)
        main_toolbar.addAction(tabbed_action)
        main_toolbar.addAction(close_action)
        main_toolbar.setAllowedAreas(Qt.TopToolBarArea | Qt.BottomToolBarArea)
        main_toolbar.setObjectName("main_toolbar")

        settings = QSettings()
        settings.beginGroup("main_window")
        self.restoreGeometry(settings.value("geometry"))
        self.restoreState(settings.value("state"))
        self.recent_files = settings.value("recent_files")
        if self.recent_files is None:
            self.recent_files = []
        elif not isinstance(self.recent_files, list):
            self.recent_files = [self.recent_files]
        self.update_recent()
        settings.endGroup()

        prev_action.setEnabled(False)
        next_action.setEnabled(False)
        tile_action.setEnabled(False)
        cascade_action.setEnabled(False)
        close_action.setEnabled(False)
        tabbed_action.setEnabled(False)
        self.tree_widget.setEnabled(False)
        self.showNormal()
        self.normal_action.setEnabled(False)
        self.show_message(self.tr("Ready"))

    # --- rest of the methods remain unchanged ---
    # keyPressEvent, change_view, closeEvent, update_recent, open_recent, initialize, load_file, 
    # open_tool, disable_bold, toggle_view, show_help, show_about, show_message, generate_report
    # Copy the existing method definitions exactly as they are from your original file.
    

if __name__ == "__main__":
    application = QApplication(sys.argv)
    mainwindow = MainWindow()
    sys.exit(application.exec())
