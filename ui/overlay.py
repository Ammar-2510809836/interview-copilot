import sys
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QFrame, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QScrollArea, QSizeGrip,
                              QSystemTrayIcon, QMenu)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QFont, QCursor, QFontDatabase, QPixmap, QPainter, QColor, QPolygon, QBrush, QPen, QIcon

class WorkerSignals(QObject):
    """Signals for communicating with the UI thread from async tasks."""
    update_text = pyqtSignal(str)

class UIOverlay(QWidget):
    # Resize zone thickness in pixels
    RESIZE_MARGIN = 8
    # Loaded monospace font family name (set by _load_fonts)
    CODE_FONT = "Consolas"  # fallback if JetBrains Mono not loaded

    @classmethod
    def _load_fonts(cls):
        """Load bundled fonts into Qt using QFontDatabase."""
        font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts", "JetBrainsMono-Regular.ttf")
        if os.path.exists(font_path):
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    cls.CODE_FONT = families[0]
                    print(f"[Font] Loaded: {cls.CODE_FONT}")
                    return
        print(f"[Font] JetBrains Mono not found, using fallback: {cls.CODE_FONT}")


    def __init__(self):
        super().__init__()
        UIOverlay._load_fonts()

        # 1. Window Settings: Frameless, Always on Top, Tool window (hides from taskbar)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(320, 250)
        self.resize(500, 580)  # Initial size — user can drag to resize

        # Resize state tracking
        self._resize_dir = None
        self._resize_start_pos = None
        self._resize_start_geom = None
        # Drag state tracking
        self._drag_pos = None

        # Main Layout wrapper for the top-level widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 2. Container to hold the styled background.
        self.container = QFrame(self)
        self.container.setObjectName("MainContainer")
        self.container.setStyleSheet("""
            QFrame#MainContainer {
                background-color: rgba(20, 20, 20, 240);
                border-radius: 12px;
                border: 1px solid #444;
            }
        """)
        main_layout.addWidget(self.container)

        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(15, 10, 15, 10)

        # 3. Top Bar (Drag Handle + Close Button)
        self.top_bar = QHBoxLayout()

        self.title_label = QLabel("Interview Copilot")
        self.title_label.setStyleSheet("color: #888; font-family: Segoe UI; font-size: 11px; font-weight: bold;")
        self.top_bar.addWidget(self.title_label)

        self.top_bar.addStretch()

        self.close_btn = QPushButton("✕")
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
        self.close_btn.clicked.connect(self.hide)
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
        self.text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.text_label.setStyleSheet(f"""
            color: #e8e8e8;
            font-size: 14px;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            line-height: 1.6;
            background: transparent;
            border: none;
            padding: 2px 4px;
        """)

        self.scroll_area.setWidget(self.text_label)
        self.layout.addWidget(self.scroll_area)

        # 5. Bottom-right resize grip for discoverability
        grip_bar = QHBoxLayout()
        grip_bar.addStretch()
        grip = QSizeGrip(self)
        grip.setFixedSize(14, 14)
        grip_bar.addWidget(grip)
        self.layout.addLayout(grip_bar)

        # Signal wiring
        self.signals = WorkerSignals()
        self.signals.update_text.connect(self._set_text)
        self.show()
        self._setup_tray()

    # --- System Tray ---
    def _make_tray_icon(self) -> QIcon:
        """Draw a neon green ⚡ lightning bolt on a dark square — no external image needed."""
        size = 64
        px = QPixmap(size, size)
        px.fill(QColor(20, 20, 20, 255))  # Dark background

        painter = QPainter(px)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Neon green lightning bolt polygon (normalised to 64x64)
        bolt = QPolygon([
            QPoint(38,  4),   # top-right
            QPoint(24, 30),   # middle-left top
            QPoint(34, 30),   # middle centre
            QPoint(18, 60),   # bottom-left
            QPoint(40, 34),   # middle-right bottom
            QPoint(30, 34),   # middle centre
            QPoint(44, 10),   # right shoulder
        ])
        painter.setBrush(QBrush(QColor(0, 250, 154)))   # #00fa9a neon green
        painter.setPen(QPen(QColor(0, 200, 120), 1))
        painter.drawPolygon(bolt)
        painter.end()

        return QIcon(px)

    def _setup_tray(self):
        """Create and show the system tray icon with a right-click context menu."""
        self.tray = QSystemTrayIcon(self._make_tray_icon(), parent=self)
        self.tray.setToolTip("Interview Copilot — Running")

        menu = QMenu()
        menu.setStyleSheet("""
            QMenu { background-color:#1e1e1e; color:#eeeeee; border:1px solid #444; font-family:'Segoe UI'; font-size:13px; }
            QMenu::item:selected { background-color:#2a2a2a; }
            QMenu::separator { height:1px; background:#444; margin:4px 0; }
        """)

        show_action = menu.addAction("⚡  Show / Hide Overlay")
        show_action.triggered.connect(self._toggle_visibility)

        menu.addSeparator()

        quit_action = menu.addAction("✕  Quit Interview Copilot")
        quit_action.triggered.connect(QApplication.instance().quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

        # Show a startup balloon notification
        self.tray.showMessage(
            "Interview Copilot",
            "Running in background. Hotkeys: Ctrl+Shift+Space (trigger), Ctrl+R (regen).",
            QSystemTrayIcon.MessageIcon.Information,
            3000
        )

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _on_tray_activated(self, reason):
        """Double-click or single-click on tray icon toggles overlay."""
        if reason in (QSystemTrayIcon.ActivationReason.DoubleClick,
                      QSystemTrayIcon.ActivationReason.Trigger):
            self._toggle_visibility()

    def _get_resize_direction(self, pos):
        """Return a string like 'left', 'right', 'top', 'bottom', 'top-left', etc. or None."""
        m = self.RESIZE_MARGIN
        w, h = self.width(), self.height()
        x, y = pos.x(), pos.y()
        on_left   = x < m
        on_right  = x > w - m
        on_top    = y < m
        on_bottom = y > h - m
        if on_top and on_left:    return "top-left"
        if on_top and on_right:   return "top-right"
        if on_bottom and on_left: return "bottom-left"
        if on_bottom and on_right:return "bottom-right"
        if on_left:   return "left"
        if on_right:  return "right"
        if on_top:    return "top"
        if on_bottom: return "bottom"
        return None

    def _cursor_for_direction(self, direction):
        cursors = {
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "top-left": Qt.CursorShape.SizeFDiagCursor,
            "bottom-right": Qt.CursorShape.SizeFDiagCursor,
            "top-right": Qt.CursorShape.SizeBDiagCursor,
            "bottom-left": Qt.CursorShape.SizeBDiagCursor,
        }
        return cursors.get(direction, Qt.CursorShape.ArrowCursor)

    # --- Mouse Events ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            direction = self._get_resize_direction(event.position().toPoint())
            if direction:
                self._resize_dir = direction
                self._resize_start_pos = event.globalPosition().toPoint()
                self._resize_start_geom = self.geometry()
            else:
                self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        global_pos = event.globalPosition().toPoint()

        if self._resize_dir and self._resize_start_pos:
            # Resizing
            delta = global_pos - self._resize_start_pos
            g = QRect(self._resize_start_geom)
            d = self._resize_dir
            if "right" in d:  g.setRight(g.right() + delta.x())
            if "bottom" in d: g.setBottom(g.bottom() + delta.y())
            if "left" in d:   g.setLeft(g.left() + delta.x())
            if "top" in d:    g.setTop(g.top() + delta.y())
            if g.width() >= self.minimumWidth() and g.height() >= self.minimumHeight():
                self.setGeometry(g)
        elif self._drag_pos:
            # Moving
            delta = global_pos - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = global_pos
        else:
            # Update cursor based on hover position
            direction = self._get_resize_direction(pos)
            if direction:
                self.setCursor(self._cursor_for_direction(direction))
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        self._resize_dir = None
        self._resize_start_pos = None
        self._resize_start_geom = None
        self._drag_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # --- Methods to Update the UI ---
    def update_text(self, text):
        """Thread safe command to update the text."""
        self.signals.update_text.emit(text)

    def _set_text(self, text):
        self.text_label.setText(text)
        vsb = self.scroll_area.verticalScrollBar()
        vsb.setValue(vsb.maximum())
        self.show()

    def clear_text(self):
        self.text_label.setText("...")

def create_app():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    return app


