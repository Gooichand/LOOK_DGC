from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton


def show_help_dialog(parent=None):
    dialog = QDialog(parent)
    dialog.setWindowTitle("LOOK-DGC Help")
    dialog.resize(700, 500)
    
    layout = QVBoxLayout()
    
    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    browser.setHtml("""
    <h2>🔍 LOOK-DGC - Digital Image Forensics Toolkit</h2>
    
    <h3>📋 Getting Started</h3>
    <ol>
        <li><b>Load Image:</b> File → Load image (Ctrl+O)</li>
        <li><b>Select Tool:</b> Double-click any tool in the left panel</li>
        <li><b>Analyze:</b> Adjust parameters and examine results</li>
    </ol>
    
    <h3>🛠️ Tool Categories</h3>
    <ul>
        <li><b>General:</b> Original image, file digest, hex editor, reverse search</li>
        <li><b>Metadata:</b> EXIF data, thumbnails, geolocation</li>
        <li><b>Inspection:</b> Magnifier, histogram, adjustments, comparison</li>
        <li><b>Color Analysis:</b> RGB/HSV plots, color spaces, PCA</li>
        <li><b>Noise Analysis:</b> Noise separation, deviation, bit planes</li>
        <li><b>JPEG Analysis:</b> Quality estimation, error level, compression</li>
        <li><b>Tampering Detection:</b> Copy-move, splicing, resampling</li>
        <li><b>AI Solutions:</b> TruFor deep learning detection</li>
    </ul>
    
    <h3>⌨️ Keyboard Shortcuts</h3>
    <ul>
        <li><b>Ctrl+O:</b> Load image</li>
        <li><b>Tab:</b> Toggle tools panel</li>
        <li><b>F1:</b> Show help</li>
        <li><b>F10:</b> Toggle tabbed view</li>
        <li><b>F11:</b> Tile windows</li>
        <li><b>F12:</b> Cascade windows</li>
    </ul>
    
    <h3>📚 Documentation</h3>
    <p>For detailed information:</p>
    <ul>
        <li><a href="https://github.com/Gooichand/LOOK-DGC/blob/main/README.md">README - Installation & Features</a></li>
        <li><a href="https://github.com/Gooichand/LOOK-DGC/blob/main/HOW_IT_WORKS.md">HOW IT WORKS - Analysis Guide</a></li>
        <li><a href="https://github.com/Gooichand/LOOK-DGC">GitHub Repository</a></li>
    </ul>
    
    <h3>💡 Tips</h3>
    <ul>
        <li>Compare original with analysis results side-by-side</li>
        <li>Use multiple tools for comprehensive analysis</li>
        <li>Export results for documentation</li>
        <li>Check metadata before pixel-level analysis</li>
    </ul>
    """)
    
    layout.addWidget(browser)
    
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.accept)
    layout.addWidget(close_btn)
    
    dialog.setLayout(layout)
    dialog.exec()
