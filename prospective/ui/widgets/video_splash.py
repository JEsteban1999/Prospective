"""Video splash screen — plays an intro clip before the login dialog.

Shows a centred, fixed-size frameless window (~900 × 506 px) that plays
the first few seconds of the intro video and then closes automatically
with a smooth **fade-out** transition.

Frames are decoded by the imageio-ffmpeg bundled binary — no DirectShow
codecs required on Windows.

Usage::

    from prospective.ui.widgets.video_splash import VideoSplashDialog
    VideoSplashDialog("resources/shutterstock_1090107869.mp4").exec()

Closes (with fade-out) when:
  • MAX_DURATION_MS elapsed (default 3 500 ms)
  • Video reaches its natural end (if shorter than max)
  • User clicks or presses any key (instant close, no fade)
"""
from __future__ import annotations

import logging
import os

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QSizePolicy, QVBoxLayout

logger = logging.getLogger(__name__)

# Maximum playback time before fade-out begins
MAX_DURATION_MS  = 5_000
# Duration of the fade-out animation
FADE_OUT_MS      = 500

_W, _H = 900, 506   # 16:9 splash size


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _imageio_available() -> bool:
    try:
        import imageio  # noqa: F401
        import numpy    # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_path(video_path: str) -> str | None:
    """Prefer .mp4 sibling over .mov; return None if nothing found."""
    abs_path = os.path.abspath(video_path)
    if not abs_path.lower().endswith(".mp4"):
        mp4 = os.path.splitext(abs_path)[0] + ".mp4"
        if os.path.isfile(mp4):
            return mp4
    if os.path.isfile(abs_path):
        return abs_path
    return None


# ──────────────────────────────────────────────────────────────────────────── #
# Splash dialog                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

class VideoSplashDialog(QDialog):
    """
    Centred frameless splash that plays *video_path* for up to
    MAX_DURATION_MS ms, then fades out gracefully before closing.
    """

    def __init__(self, video_path: str, parent=None) -> None:
        super().__init__(parent)

        self._reader    = None
        self._timer     = None
        self._max_timer = None
        self._anim      = None   # keep reference so GC doesn't kill it
        self._closed    = False
        self._frame_ms  = 40

        # ── window style ──────────────────────────────────────────────── #
        self.setWindowFlags(
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setFixedSize(_W, _H)
        self.setStyleSheet("background: black;")
        self.setWindowOpacity(0.0)   # start invisible — fade IN first

        # Centre on primary screen
        screen = QApplication.primaryScreen().geometry()
        self.move(
            screen.center().x() - _W // 2,
            screen.center().y() - _H // 2,
        )

        # ── video label ───────────────────────────────────────────────── #
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet("background: black;")
        self._lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._lbl)

        # ── hard-stop timer (starts after fade-in completes) ──────────── #
        self._max_timer = QTimer(self)
        self._max_timer.setSingleShot(True)
        self._max_timer.timeout.connect(self._fade_and_close)

        # ── start playback ────────────────────────────────────────────── #
        playable = _resolve_path(video_path)
        if playable and _imageio_available():
            self._fade_in(playable)
        else:
            if not playable:
                logger.warning("VideoSplashDialog: file not found — %s", video_path)
            if not _imageio_available():
                logger.warning("VideoSplashDialog: imageio not available — skipping.")
            QTimer.singleShot(200, self._close)

    # ------------------------------------------------------------------ #
    # Fade-in on open                                                     #
    # ------------------------------------------------------------------ #

    def _fade_in(self, path: str) -> None:
        """Fade window in (0→1) over 400 ms, then start playback."""
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(400)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.finished.connect(lambda: self._start(path))
        self._anim = anim
        anim.start()

    # ------------------------------------------------------------------ #
    # Playback                                                             #
    # ------------------------------------------------------------------ #

    def _start(self, path: str) -> None:
        try:
            import imageio
            self._reader = imageio.get_reader(path)
            meta = self._reader.get_meta_data()
            fps  = float(meta.get("fps") or 50.0)
            self._frame_ms = max(1, int(1000.0 / fps))
            logger.info("VideoSplashDialog: %s  %.0f fps", os.path.basename(path), fps)
        except Exception as exc:
            logger.warning("VideoSplashDialog: cannot open reader — %s", exc)
            self._fade_and_close()
            return

        self._iter  = iter(self._reader)
        self._timer = QTimer(self)
        self._timer.setInterval(self._frame_ms)
        self._timer.timeout.connect(self._next_frame)
        self._timer.start()

        # Start the max-duration countdown *after* fade-in
        self._max_timer.start(MAX_DURATION_MS)

    def _next_frame(self) -> None:
        try:
            frame = next(self._iter)
        except StopIteration:
            self._fade_and_close()
            return
        except Exception as exc:
            logger.warning("VideoSplashDialog: frame error — %s", exc)
            self._fade_and_close()
            return
        self._show_frame(frame)

    def _show_frame(self, frame) -> None:
        import numpy as np
        from PyQt5.QtGui import QImage, QPixmap

        h, w, ch = frame.shape
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        img = QImage(frame.data, w, h, w * ch, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            _W, _H,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._lbl.setPixmap(pix)

    # ------------------------------------------------------------------ #
    # Fade-out on close                                                    #
    # ------------------------------------------------------------------ #

    def _fade_and_close(self) -> None:
        """Pause playback, animate opacity 1→0, then accept."""
        if self._closed:
            return
        # Stop frame timer so the image freezes during fade-out
        if self._timer:
            self._timer.stop()
        if self._max_timer:
            self._max_timer.stop()

        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(FADE_OUT_MS)
        anim.setStartValue(self.windowOpacity())
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(self._close)
        self._anim = anim   # keep reference
        anim.start()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass
        self.accept()

    # ------------------------------------------------------------------ #
    # User interaction — instant close (no fade, just skip)               #
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event) -> None:
        self._close()

    def keyPressEvent(self, event) -> None:
        self._close()

    def closeEvent(self, event) -> None:
        self._close()
        super().closeEvent(event)
