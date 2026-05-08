"""AvatarWidget — QPainter-based circular user icon.

Draws a classic person-silhouette (head + shoulders) clipped to a circle.
When a photo is loaded via set_photo() the image is displayed instead,
also clipped to a circle with a thin border ring.

No external dependencies beyond PyQt5.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QBrush, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QWidget


class AvatarWidget(QWidget):
    """Circular avatar that paints a person silhouette or a user photo.

    Parameters
    ----------
    size:         Diameter in pixels (widget is always square).
    bg_color:     Fill of the circle background.
    icon_color:   Fill of the head+shoulder silhouette.
    border_color: 2-px ring around the outer edge of the circle.
    """

    def __init__(
        self,
        size: int = 68,
        bg_color: str = "#2A2A2A",
        icon_color: str = "#9B9B9B",
        border_color: str = "#363636",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._s   = size
        self._bg  = QColor(bg_color)
        self._fg  = QColor(icon_color)
        self._bdr = QColor(border_color)
        self._pix: QPixmap | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def set_photo(self, path: str) -> None:
        """Load an image from *path* and display it inside the circle."""
        pix = QPixmap(path)
        if not pix.isNull():
            self._pix = pix
            self.update()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """Set a pre-loaded QPixmap directly."""
        if not pixmap.isNull():
            self._pix = pixmap
            self.update()

    def clear_photo(self) -> None:
        """Revert to the silhouette icon."""
        self._pix = None
        self.update()

    # ------------------------------------------------------------------ #
    # Painting                                                             #
    # ------------------------------------------------------------------ #

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        s    = self._s
        bdr  = 1.0           # border ring half-width (pixels)
        inner = QRectF(bdr, bdr, s - 2 * bdr, s - 2 * bdr)

        # ── Clip everything to the circle ───────────────────────────── #
        clip = QPainterPath()
        clip.addEllipse(inner)
        p.setClipPath(clip)

        if self._pix is not None:
            # Scale to fill, center-crop
            sp = self._pix.scaled(
                s, s, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            ox = (sp.width()  - s) // 2
            oy = (sp.height() - s) // 2
            p.drawPixmap(0, 0, sp, ox, oy, s, s)
        else:
            # Background fill
            p.fillRect(0, 0, s, s, self._bg)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(self._fg))

            # Head — circle, upper-centre of the avatar
            hr        = s * 0.22            # head radius  (≈15 px @ 68)
            hcx, hcy  = s * 0.50, s * 0.37
            p.drawEllipse(QRectF(hcx - hr, hcy - hr, hr * 2, hr * 2))

            # Shoulders — large ellipse rising from below the frame
            srx, sry  = s * 0.40, s * 0.34  # semi-axes
            scx, scy  = s * 0.50, s * 0.96  # centre (mostly below visible area)
            p.drawEllipse(QRectF(scx - srx, scy - sry, srx * 2, sry * 2))

        # ── Border ring (drawn outside the clip) ────────────────────── #
        p.setClipping(False)
        p.setPen(QPen(self._bdr, 2.0))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(inner)

        p.end()
