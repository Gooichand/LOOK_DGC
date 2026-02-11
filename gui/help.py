from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
from PySide6.QtCore import Qt


def show_help_dialog(parent):
    dialog = QDialog(parent)
    dialog.setWindowTitle("LOOK-DGC Help")
    dialog.resize(700, 500)
    
    layout = QVBoxLayout()
    
    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    browser.setHtml("""
    <h2>LOOK-DGC - Digital Image Forensics Toolkit</h2>
    <p>Quick reference guide for analyzing images.</p>
    
    <h3>📋 General Tools</h3>
    <ul>
        <li><b>Original Image</b>: View unaltered image</li>
        <li><b>File Digest</b>: Cryptographic hashes</li>
        <li><b>Hex Editor</b>: Raw byte examination</li>
        <li><b>Similar Search</b>: Reverse image search</li>
    </ul>
    
    <h3>📊 Metadata Tools</h3>
    <ul>
        <li><b>EXIF Full Dump</b>: Complete metadata extraction</li>
        <li><b>Thumbnail Analysis</b>: Embedded thumbnail comparison</li>
        <li><b>Geolocation Data</b>: GPS coordinate mapping</li>
    </ul>
    
    <h3>🔬 Inspection Tools</h3>
    <ul>
        <li><b>Enhancing Magnifier</b>: Forgery-detection magnification</li>
        <li><b>Channel Histogram</b>: RGB/composite analysis</li>
        <li><b>Reference Comparison</b>: Dual-image comparison</li>
    </ul>
    
    <h3>⚠️ Tampering Detection</h3>
    <ul>
        <li><b>Copy-Move Forgery</b>: Cloning detection</li>
        <li><b>Composite Splicing</b>: Splicing detection</li>
        <li><b>Image Resampling</b>: Interpolation traces</li>
    </ul>
    
    <p><b>Full Documentation:</b> <a href="https://github.com/Gooichand/LOOK-DGC">GitHub Repository</a></p>
    <p><b>Keyboard Shortcuts:</b> F1 = Help | Ctrl+O = Open Image</p>
    """)
    layout.addWidget(browser)
    
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.accept)
    layout.addWidget(close_btn)
    
    dialog.setLayout(layout)
    dialog.exec()
