import sys
import os
import re
import ctypes
import ctypes.wintypes
import logging
from PyQt6.QtWidgets import (QApplication, QWidget, QFrame, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QScrollArea, QSizeGrip, QTextEdit,
                              QSystemTrayIcon, QMenu)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QPoint, QRect, QEvent, QTimer
from PyQt6.QtGui import QFont, QCursor, QFontDatabase, QPixmap, QPainter, QColor, QPolygon, QBrush, QPen, QIcon

overlay_logger = logging.getLogger(__name__)

class WorkerSignals(QObject):
    """Signals for communicating with the UI thread from async tasks."""
    update_text = pyqtSignal(str, str, int)
    set_typing_indicator = pyqtSignal(bool)
    manual_question_submitted = pyqtSignal(str)

class UIOverlay(QWidget):
    # Resize zone thickness in pixels
    RESIZE_MARGIN = 8
    # Loaded monospace font family name (set by _load_fonts)
    CODE_FONT = "Consolas"  # fallback if JetBrains Mono not loaded

    # STAR section colors (subtle left borders)
    STAR_COLORS = {
        "situation": "#4a90d9",  # Blue
        "task": "#f5a623",       # Orange
        "action": "#7ed321",     # Green
        "result": "#bd10e0",     # Purple
    }

    @classmethod
    def _load_fonts(cls):
        """Load bundled fonts into Qt using QFontDatabase. Supports PyInstaller sys._MEIPASS."""
        import sys # ensure sys is available here just in case, though it's at the top of the file
        if getattr(sys, 'frozen', False):
            # If the application is run as a bundle, the PyInstaller bootloader
            # extends the sys module by a flag frozen=True and sets the app
            # path into variable _MEIPASS.
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(__file__))

        font_path = os.path.join(base_path, "assets", "fonts", "JetBrainsMono-Regular.ttf")
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

        self.close_btn = QPushButton("")
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
        self.scroll_area.setMinimumHeight(140)
        self.layout.addWidget(self.scroll_area)

        # Typing indicator label (hidden by default)
        self.typing_indicator = QLabel("")
        self.typing_indicator.setStyleSheet("""
            color: #00fa9a;
            font-size: 12px;
            font-family: 'Segoe UI', sans-serif;
            font-style: italic;
            padding: 4px 0;
        """)
        self.typing_indicator.hide()
        self.layout.addWidget(self.typing_indicator)

        # 5. Manual pasted-question input
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Paste an interview question...")
        self.chat_input.setFixedHeight(64)
        self.chat_input.installEventFilter(self)
        self.chat_input.setStyleSheet(f"""
            QTextEdit {{
                color: #eeeeee;
                background-color: rgba(35, 35, 35, 230);
                border: 1px solid #444;
                border-radius: 6px;
                padding: 6px 8px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 13px;
                selection-background-color: #00a866;
            }}
            QTextEdit:focus {{
                border: 1px solid #00fa9a;
            }}
        """)
        self.layout.addWidget(self.chat_input)

        chat_bar = QHBoxLayout()
        chat_bar.addStretch()
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedSize(70, 28)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet("""
            QPushButton {
                color: #101010;
                background-color: #00fa9a;
                border-radius: 6px;
                font-weight: bold;
                font-family: 'Segoe UI';
                border: none;
            }
            QPushButton:hover { background-color: #22ffad; }
            QPushButton:pressed { background-color: #00c77a; }
        """)
        self.send_btn.clicked.connect(self._submit_manual_question)
        chat_bar.addWidget(self.send_btn)
        self.layout.addLayout(chat_bar)

        # 6. Bottom-right resize grip for discoverability
        grip_bar = QHBoxLayout()
        grip_bar.addStretch()
        grip = QSizeGrip(self)
        grip.setFixedSize(14, 14)
        grip_bar.addWidget(grip)
        self.layout.addLayout(grip_bar)

        # Signal wiring
        self.signals = WorkerSignals()
        self.signals.update_text.connect(self._set_text)
        self.signals.set_typing_indicator.connect(self._set_typing_indicator)
        self.show()
        self._apply_capture_exclusion()
        self._setup_tray()

    def _apply_capture_exclusion(self):
        """
        Makes this window invisible to ALL screen capture software:
        - Screen sharing (micro1, Zoom, Teams, Google Meet)
        - Screen recording (OBS, Windows Game Bar, ShareX)
        - PrintScreen / Snipping Tool

        Uses Windows API: SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        WDA_EXCLUDEFROMCAPTURE = 0x00000011

        The window remains fully visible to the user on their physical monitor.
        Only works on Windows 10 Build 2004+ and Windows 11.
        """
        try:
            # Get the native Windows HWND from Qt
            hwnd = int(self.winId())
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            result = ctypes.windll.user32.SetWindowDisplayAffinity(
                ctypes.wintypes.HWND(hwnd),
                ctypes.wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE)
            )
            if result:
                overlay_logger.info("[STEALTH] Window is now INVISIBLE to all screen capture and recording software.")
            else:
                error_code = ctypes.GetLastError()
                overlay_logger.warning(f"[STEALTH] SetWindowDisplayAffinity failed (error={error_code}). Requires Windows 10 Build 2004+.")
        except Exception as e:
            overlay_logger.warning(f"[STEALTH] Could not apply capture exclusion: {e}")

    # --- Structured Answer Formatting ---
    def format_structured_answer(self, answer_text: str, question_text: str, question_type: str = "generic") -> str:
        """
        Parse different answer formats (STAR, technical, code) and apply appropriate HTML styling.

        Args:
            answer_text: The raw answer text from the LLM or pre-formatted HTML
            question_type: Type of question - "star", "technical", "code", or "generic"

        Returns:
            HTML formatted string ready for display
        """
        if not answer_text:
            return answer_text

        # Check if text is already HTML (starts with < and contains HTML tags)
        # Using a more robust check for HTML tags
        is_already_html = answer_text.strip().startswith('<') and any(tag in answer_text.lower() for tag in ['<div', '<span', '<br', '<b>', '<i', '<p', '<code'])

        if is_already_html:
            # Already HTML - don't escape, just return as-is
            return answer_text

        # Escape HTML special characters first for raw text
        import html
        escaped_text = html.escape(answer_text)

        # Convert newlines to <br> tags
        escaped_text = escaped_text.replace("\n", "<br>")

        formatted_html = escaped_text

        if question_type.lower() == "star":
            formatted_html = self._format_star_answer(formatted_html)
        elif question_type.lower() in ["technical", "code"]:
            formatted_html = self._format_code_blocks(formatted_html)
        else:
            # Generic formatting - just detect and format code blocks
            formatted_html = self._format_code_blocks(formatted_html)

        formatted_html = self._format_inline_markdown(formatted_html)
        formatted_html = self._format_bullet_lines(formatted_html)

        return formatted_html

    def _format_star_answer(self, html_text: str) -> str:
        """Format STAR method answers with highlighted sections."""
        # Pattern to match STAR headers (case insensitive)
        star_headers = [
            (r'(?i)(Situation|S|## Situation)', 'situation'),
            (r'(?i)(Task|T|## Task)', 'task'),
            (r'(?i)(Action|A|## Action)', 'action'),
            (r'(?i)(Result|R|## Result)', 'result'),
        ]

        for pattern, section_type in star_headers:
            # Replace headers with styled versions
            def replace_header(match):
                header_text = match.group(1).replace('## ', '')
                color = self.STAR_COLORS.get(section_type, "#888")
                return f'<div style="margin-top: 12px; padding: 8px 12px; border-left: 3px solid {color}; background-color: rgba({self._hex_to_rgb(color)}, 0.08); border-radius: 0 4px 4px 0;"><span style="color: {color}; font-weight: bold; font-size: 13px;">{header_text}</span></div>'

            html_text = re.sub(pattern + r'[:\s]*(?:<br>|\n|$)', replace_header, html_text)

        return html_text

    def _format_code_blocks(self, html_text: str) -> str:
        """Detect and format code blocks in markdown style."""
        # Pattern to match ```language\ncode\n``` blocks
        code_pattern = r'```(\w+)?<br>(.*?)<br>```'

        def replace_code_block(match):
            language = match.group(1) or "python"
            code = match.group(2)
            return self.render_code_block(code, language)

        html_text = re.sub(code_pattern, replace_code_block, html_text, flags=re.DOTALL)

        # Also handle inline `code` with backticks
        inline_pattern = r'`([^`]+)`'
        html_text = re.sub(inline_pattern,
                          r'<code style="background-color: #2d2d2d; color: #00fa9a; padding: 1px 4px; border-radius: 3px; font-family: \'{}\', monospace; font-size: 13px;">\1</code>'.format(self.CODE_FONT),
                          html_text)

        return html_text

    def _format_inline_markdown(self, html_text: str) -> str:
        """Apply lightweight markdown emphasis after HTML escaping."""
        html_text = re.sub(r'\*\*(.+?)\*\*', r'<b style="color:#ffffff;">\1</b>', html_text)
        html_text = re.sub(r'(?<!\*)\*([^*<][^*]*?)\*(?!\*)', r'<i>\1</i>', html_text)
        return html_text

    def _format_bullet_lines(self, html_text: str) -> str:
        """Turn common markdown bullet lines into stable Qt HTML rows."""
        lines = html_text.split("<br>")
        formatted = []
        bullet_prefixes = ("- ", "* ", "• ", "▸ ", "&#8226; ", "&#9656; ")

        for line in lines:
            stripped = line.strip()
            matched_prefix = next((prefix for prefix in bullet_prefixes if stripped.startswith(prefix)), None)
            if matched_prefix:
                content = stripped[len(matched_prefix):].strip()
                formatted.append(
                    "<div style='margin:2px 0 2px 8px; color:#cccccc;'>"
                    "<span style='color:#00fa9a; font-weight:bold;'>&rsaquo;</span>&nbsp;"
                    f"{content}</div>"
                )
            elif not stripped:
                formatted.append("<div style='height:4px;'></div>")
            else:
                formatted.append(f"{line}<br>")

        return "".join(formatted)

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string for rgba()."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}"

    def render_star_section(self, content: str, section_type: str) -> str:
        """
        Render individual STAR sections with color-coded left border.

        Args:
            content: The section content text
            section_type: One of: "situation", "task", "action", "result"

        Returns:
            HTML formatted section
        """
        section_type = section_type.lower()
        color = self.STAR_COLORS.get(section_type, "#888")
        title = section_type.capitalize()

        rgb = self._hex_to_rgb(color)

        html = f"""
        <div style="margin: 10px 0; padding: 12px 16px; border-left: 4px solid {color};
                    background-color: rgba({rgb}, 0.08); border-radius: 0 6px 6px 0;">
            <div style="color: {color}; font-weight: bold; font-size: 13px; margin-bottom: 6px;">
                {title}
            </div>
            <div style="color: #e8e8e8; line-height: 1.5;">
                {content}
            </div>
        </div>
        """
        return html.strip()

    def render_code_block(self, code: str, language: str = "python") -> str:
        """
        Return HTML formatted code block with syntax highlighting styling.
        """
        import html
        # Step 1: Unescape HTML entities (e.g. &amp; &lt; &gt;)
        code = html.unescape(code)
        # Step 2: Normalize <br> tags back to real newlines
        # This is critical — the pipeline converts \n→<br> before code is extracted,
        # so we must reverse that before passing to syntax highlighter.
        code = code.replace("<br>", "\n").replace("<br/>", "\n").replace("<BR>", "\n")

        # Step 3: Apply syntax highlighting (works on clean newline-separated code)
        highlighted = self._apply_syntax_highlighting(code, language)

        # Step 4: Convert newlines back to <br> for Qt's HTML renderer
        highlighted = highlighted.replace("\n", "<br>")

        html_block = f"""
        <div style="margin: 12px 0; background-color: #1e1e1e; border: 1px solid #333;
                    border-radius: 8px; overflow: hidden;">
            <div style="background-color: #2d2d2d; padding: 6px 12px; font-size: 11px;
                        color: #888; font-family: '{self.CODE_FONT}', monospace; border-bottom: 1px solid #333;">
                {language.upper()}
            </div>
            <div style="padding: 12px; background-color: #1e1e1e; font-family: '{self.CODE_FONT}', monospace;
                        font-size: 13px; line-height: 1.8; color: #d4d4d4;">
                {highlighted}
            </div>
        </div>
        """
        return html_block.strip()

    def _apply_syntax_highlighting(self, code: str, language: str) -> str:
        """Apply basic syntax highlighting to code."""
        import html

        # Keywords for Python
        if language.lower() == "python":
            keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return',
                       'import', 'from', 'as', 'try', 'except', 'finally', 'with', 'pass',
                       'break', 'continue', 'lambda', 'yield', 'raise', 'assert', 'del',
                       'global', 'nonlocal', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None']

            # Escape HTML
            code = html.escape(code)

            # Highlight keywords
            for kw in keywords:
                pattern = r'\b' + kw + r'\b'
                code = re.sub(pattern, f'<span style="color: #569cd6;">{kw}</span>', code)

            # Highlight strings (simple pattern)
            code = re.sub(r'(".*?")', r'<span style="color: #ce9178;">\1</span>', code)
            code = re.sub(r"('.*?')", r'<span style="color: #ce9178;">\1</span>', code)

            # Highlight comments
            code = re.sub(r'(#.*)$', r'<span style="color: #6a9955;">\1</span>', code, flags=re.MULTILINE)

            # Highlight functions
            code = re.sub(r'(\w+)(\s*\()', r'<span style="color: #dcdcaa;">\1</span>\2', code)

        elif language.lower() in ["javascript", "js", "typescript", "ts"]:
            keywords = ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while',
                       'return', 'import', 'export', 'from', 'class', 'new', 'this', 'try',
                       'catch', 'finally', 'async', 'await', 'true', 'false', 'null', 'undefined']

            code = html.escape(code)

            for kw in keywords:
                pattern = r'\b' + kw + r'\b'
                code = re.sub(pattern, f'<span style="color: #569cd6;">{kw}</span>', code)

            code = re.sub(r'(".*?")', r'<span style="color: #ce9178;">\1</span>', code)
            code = re.sub(r"('.*?')", r'<span style="color: #ce9178;">\1</span>', code)
            code = re.sub(r'(`.*?`)', r'<span style="color: #ce9178;">\1</span>', code)
            code = re.sub(r'(//.*)$', r'<span style="color: #6a9955;">\1</span>', code, flags=re.MULTILINE)

        else:
            # Generic highlighting
            code = html.escape(code)
            code = re.sub(r'(".*?")', r'<span style="color: #ce9178;">\1</span>', code)
            code = re.sub(r"('.*?')", r'<span style="color: #ce9178;">\1</span>', code)

        return code

    # --- System Tray ---
    def _make_tray_icon(self) -> QIcon:
        """Draw a neon green  lightning bolt on a dark square — no external image needed."""
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

        show_action = menu.addAction("  Show / Hide Overlay")
        show_action.triggered.connect(self._toggle_visibility)

        menu.addSeparator()

        quit_action = menu.addAction("  Quit Interview Copilot")
        quit_action.triggered.connect(QApplication.instance().quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

        # Show a startup balloon notification
        self.tray.showMessage(
            "Interview Copilot",
            "Running in background. Hotkeys: Ctrl+Shift+Space (trigger), Ctrl+R (regen). Paste questions in the overlay.",
            QSystemTrayIcon.MessageIcon.Information,
            3000
        )

    def eventFilter(self, obj, event):
        if obj is self.chat_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self._submit_manual_question()
                    return True
        return super().eventFilter(obj, event)

    def _submit_manual_question(self):
        question = self.chat_input.toPlainText().strip()
        if not question:
            return
        self.chat_input.clear()
        self.signals.manual_question_submitted.emit(question)

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
    def set_typing_indicator(self, visible: bool):
        """Show/hide a subtle 'Copilot is thinking...' indicator."""
        self.signals.set_typing_indicator.emit(visible)

    def _set_typing_indicator(self, visible: bool):
        """Internal method to update typing indicator visibility."""
        if visible:
            self.typing_indicator.setText("Copilot is thinking...")
            self.typing_indicator.show()
        else:
            self.typing_indicator.hide()

    def update_text(self, text: str, question_type: str = "generic", is_streaming: bool = False, scroll_mode: str = None):
        """
        Thread-safe command to update the text with optional formatting.

        Args:
            text: The text content to display
            question_type: Type of question ("star", "technical", "code", "generic")
            is_streaming: If True, show ' Generating...' indicator
            scroll_mode: "top", "preserve", or "bottom". Defaults to preserving during streaming
                and showing the start of the answer for completed renders.
        """
        formatted_text = self.format_structured_answer(text, "", question_type)

        if is_streaming:
            formatted_text += '<div style="color: #00fa9a; font-style: italic; margin-top: 8px; font-size: 12px;"> Generating...</div>'

        if scroll_mode is None:
            scroll_mode = "preserve" if is_streaming else "top"

        saved_scroll = self.scroll_area.verticalScrollBar().value()
        self.signals.update_text.emit(formatted_text, scroll_mode, saved_scroll)

    def _set_text(self, text, scroll_mode: str = "top", saved_scroll: int = 0):
        self.text_label.setText(text)
        self.text_label.adjustSize()
        QTimer.singleShot(0, lambda: self._apply_scroll_mode(scroll_mode, saved_scroll))
        self.show()

    def _apply_scroll_mode(self, scroll_mode: str, saved_scroll: int = 0):
        vsb = self.scroll_area.verticalScrollBar()
        if scroll_mode == "bottom":
            vsb.setValue(vsb.maximum())
        elif scroll_mode == "preserve":
            vsb.setValue(min(saved_scroll, vsb.maximum()))
        else:
            vsb.setValue(0)

    def clear_text(self):
        self.text_label.setText("...")

def create_app():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    return app
