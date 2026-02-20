import sys
from PyQt6.QtWidgets import QApplication, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QPoint, QTimer
from PyQt6.QtGui import QFont

class WorkerSignals(QObject):
    """Signals for communicating with the UI thread from async tasks."""
    update_text = pyqtSignal(str)

class UIOverlay(QWidget):
    def __init__(self):
        super().__init__()
        
        # 1. Window Settings: Frameless, Always on Top, Tool window (hides from taskbar)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Main Layout wrapper for the top-level widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 2. Container to hold the styled background. 
        self.container = QFrame(self)
        self.container.setObjectName("MainContainer")
        self.container.setFixedSize(450, 550) # Fixed Final Round AI teleprompter size
        self.container.setStyleSheet("""
            QFrame#MainContainer {
                background-color: rgba(20, 20, 20, 240); 
                border-radius: 12px;
                border: 1px solid #444;
            }
        """)
        main_layout.addWidget(self.container)

        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(15, 10, 15, 15)

        # 3. Top Bar (Drag Handle + Close Button)
        self.top_bar = QHBoxLayout()
        
        self.title_label = QLabel("Interview Copilot")
        self.title_label.setStyleSheet("color: #888; font-family: Segoe UI; font-size: 11px; font-weight: bold;")
        self.top_bar.addWidget(self.title_label)
        
        self.top_bar.addStretch()  # Pushes the button to the far right
        
        self.close_btn = QPushButton("X")
        self.close_btn.setFixedSize(22, 22)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setStyleSheet("""
            QPushButton {
                color: white; 
                background-color: #ff4747; 
                border-radius: 11px; 
                font-weight: bold;
                font-family: Arial;
                border: none;
            }
            QPushButton:hover { background-color: #ff1a1a; }
        """)
        self.close_btn.clicked.connect(self.hide) # Hides the window when clicked
        self.top_bar.addWidget(self.close_btn)
        
        self.layout.addLayout(self.top_bar)

        # 4. Scrollable Text Display Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #2a2a2a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        self.text_label = QLabel("Waiting for conversation...")
        self.text_label.setWordWrap(True)
        self.text_label.setTextFormat(Qt.TextFormat.RichText)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.text_label.setStyleSheet("""
            color: #eeeeee; 
            font-size: 15px; 
            font-family: Segoe UI, sans-serif;
            background: transparent;
            border: none;
        """)
        
        self.scroll_area.setWidget(self.text_label)
        self.layout.addWidget(self.scroll_area)

        # Variables for dragging the frameless window
        self._old_pos = None
        
        # Signal wiring for thread safety
        self.signals = WorkerSignals()
        self.signals.update_text.connect(self._set_text)
        self.show()

    # --- Methods to Update the UI ---
    def update_text(self, text):
        """Thread safe command to update the text."""
        self.signals.update_text.emit(text)
        
    def _set_text(self, text):
        """Call this method from your main loop to update the text."""
        self.text_label.setText(text)
        
        # Auto-scroll to the bottom when new text arrives
        vsb = self.scroll_area.verticalScrollBar()
        vsb.setValue(vsb.maximum())
        
        self.show() # Automatically pop back up if it was hidden

    def clear_text(self):
        self.text_label.setText("...")

    # --- Make the Frameless Window Draggable ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._old_pos:
            delta = event.globalPosition().toPoint() - self._old_pos
            self.move(self.pos() + delta)
            self._old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._old_pos = None

def create_app():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    return app
