"""Monochrome icon constants for the Prospective UI.

WHY THIS FILE EXISTS
--------------------
Qt on Windows renders Unicode characters above U+2FFF through the system
emoji font (Segoe UI Emoji), which uses COLR colour glyphs that **ignore**
the CSS ``color`` property.

Two techniques force monochrome rendering:

1. **BMP-only characters** (U+0000–U+FFFF) from the mathematical and
   geometric Unicode blocks are looked up in Segoe UI Symbol before the
   emoji font, so they always render in the widget's text colour.

2. **Variation Selector-15** (U+FE0E) appended to Miscellaneous-Symbol /
   Dingbat characters explicitly requests *text* (monochrome) presentation
   on any compliant Unicode renderer.

USAGE
-----
    from prospective.ui.icons import I
    btn = QPushButton(I.CLIPS)
    btn.setStyleSheet(f"font-family: {I.FONT}; font-size: 16px;")
"""

from __future__ import annotations

# Variation Selector-15: force text/monochrome rendering
_T: str = "︎"

# Font stack: look up glyphs in the monochrome symbol font first.
# Qt applies this per-widget via the font-family CSS property.
FONT: str = (
    "'Segoe UI Symbol', 'Segoe UI', 'Arial Unicode MS', "
    "'DejaVu Sans', sans-serif"
)


# ──────────────────────────────────────────────────────────────────────────── #
# Workflow steps                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #
#   Shown in the WorkflowStepper bar across the top of the main window.
#   Use BMP geometric / mathematical glyphs — guaranteed monochrome.

STEP_PATIENT: str = "◻"          # ◻  medium white square     (patient)
STEP_SEGMENT: str = "⊕"          # ⊕  circled plus            (grow/segment)
STEP_DETECT:  str = "◎"          # ◎  bullseye                (find candidates)
STEP_MORPHO:  str = "∑"          # ∑  n-ary summation         (analysis)
STEP_PLAN:    str = "✛"          # ✛  open-centre cross       (surgical plan)
STEP_EXPORT:  str = "⊟"          # ⊟  squared minus           (export/package)


# ──────────────────────────────────────────────────────────────────────────── #
# Navigation & global actions                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

HOME:    str = "⌂"               # ⌂  house
EDIT:    str = "✏" + _T          # ✏  pencil
FOLDER:  str = "⊡"               # ⊡  squared dot (open / load)
DOC:     str = "≡"               # ≡  identical-to (document lines)
SAVE:    str = "⊞"               # ⊞  squared plus
PRINT:   str = "⊡"               # ⊡  squared dot
ATTACH:  str = "⊕"               # ⊕  circled plus
LINK:    str = "⊕"               # ⊕  circled plus
REFRESH: str = "↻"               # ↻  clockwise open-circle arrow
LOCK:    str = "⊟"               # ⊟  squared minus
SEARCH:  str = "◎"               # ◎  bullseye
AUDIT:   str = "⊟"               # ⊟  squared minus

USER:    str = "◻"               # ◻  white square  (person placeholder)
USERS:   str = "≡"               # ≡  identical-to  (multiple lines)


# ──────────────────────────────────────────────────────────────────────────── #
# 3-D Planning sidebar buttons                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

CLIPS:      str = "✂" + _T       # ✂  scissors
FLOW_DIV:   str = "⊛"            # ⊛  circled-asterisk  (mesh/flow)
STENT:      str = "⚕" + _T       # ⚕  staff of Aesculapius
STENT_CL:   str = "⚕" + _T       # ⚕  (stent on centre-line)
TRAJECTORY: str = "⌖"            # ⌖  position indicator (crosshair)
CENTERLINE: str = "∿"            # ∿  sine wave  (winding vessel)
MEASURE:    str = "↔"            # ↔  left-right arrow
CUT:        str = "✂" + _T       # ✂  scissors
SETTINGS:   str = "⚙" + _T       # ⚙  gear


# ──────────────────────────────────────────────────────────────────────────── #
# Medical / intra-procedure actions                                              #
# ──────────────────────────────────────────────────────────────────────────── #

CLIP_PLACE:   str = "↓"          # ↓  down arrow  (place / pin)
SEED:         str = "⊙"          # ⊙  circled dot  (seed point)
GROWTH:       str = "↑"          # ↑  up arrow  (grow)
COIL:         str = "⊕"          # ⊕  circled plus  (coil/embolise)
BRAIN:        str = "✺"          # ✺  sixteen-pointed asterisk (neural)
MARK_PERF:    str = "✦"          # ✦  four-pointed star  (perforator)
ANNOTATION:   str = "⊞"          # ⊞  squared plus
ANGLE_MEAS:   str = "∠"          # ∠  angle symbol
MEDICAL_SIGN: str = "⚕" + _T     # ⚕  staff of Aesculapius


# ──────────────────────────────────────────────────────────────────────────── #
# Visibility / MPR                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

EYE:        str = "◉"            # ◉  fisheye  (visible)
EYE_HIDDEN: str = "◌"            # ◌  dotted circle  (hidden)
MPR_VIEW:   str = "⊞"            # ⊞  squared plus  (multi-planar)
OBLIQUE:    str = "◇"            # ◇  white diamond  (oblique MPR)


# ──────────────────────────────────────────────────────────────────────────── #
# Theme toggle                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #

THEME_LIGHT: str = "◐"           # ◐  left-half-black circle (bright)
THEME_DARK:  str = "◑"           # ◑  right-half-black circle (dark)


# ──────────────────────────────────────────────────────────────────────────── #
# Status / feedback                                                              #
# ──────────────────────────────────────────────────────────────────────────── #

STATUS_OK:   str = "✓" + _T      # ✓  check mark
STATUS_WARN: str = "⚠" + _T      # ⚠  warning sign
STATUS_FAIL: str = "✗" + _T      # ✗  ballot X
WAIT:        str = "↻"           # ↻  clockwise (spinner placeholder)
HINT:        str = "⊕"           # ⊕  circled plus


# ──────────────────────────────────────────────────────────────────────────── #
# Namespace alias                                                                #
#   from prospective.ui.icons import I                                           #
#   QPushButton(I.CLIPS)                                                         #
# ──────────────────────────────────────────────────────────────────────────── #

class _Icons:
    FONT         = FONT
    # Workflow
    STEP_PATIENT = STEP_PATIENT
    STEP_SEGMENT = STEP_SEGMENT
    STEP_DETECT  = STEP_DETECT
    STEP_MORPHO  = STEP_MORPHO
    STEP_PLAN    = STEP_PLAN
    STEP_EXPORT  = STEP_EXPORT
    # Navigation
    HOME     = HOME;  EDIT    = EDIT;   FOLDER  = FOLDER
    DOC      = DOC;   SAVE    = SAVE;   PRINT   = PRINT
    ATTACH   = ATTACH; LINK   = LINK;   REFRESH = REFRESH
    LOCK     = LOCK;  SEARCH  = SEARCH; AUDIT   = AUDIT
    USER     = USER;  USERS   = USERS
    # Planning sidebar
    CLIPS      = CLIPS;   FLOW_DIV  = FLOW_DIV
    STENT      = STENT;   STENT_CL  = STENT_CL
    TRAJECTORY = TRAJECTORY; CENTERLINE = CENTERLINE
    MEASURE    = MEASURE; CUT       = CUT
    SETTINGS   = SETTINGS
    # Medical actions
    CLIP_PLACE   = CLIP_PLACE;   SEED        = SEED
    GROWTH       = GROWTH;       COIL        = COIL
    BRAIN        = BRAIN;        MARK_PERF   = MARK_PERF
    ANNOTATION   = ANNOTATION;   ANGLE_MEAS  = ANGLE_MEAS
    MEDICAL_SIGN = MEDICAL_SIGN
    # Visibility
    EYE = EYE; EYE_HIDDEN = EYE_HIDDEN; MPR_VIEW = MPR_VIEW; OBLIQUE = OBLIQUE
    # Theme
    THEME_LIGHT = THEME_LIGHT; THEME_DARK = THEME_DARK
    # Status
    STATUS_OK = STATUS_OK; STATUS_WARN = STATUS_WARN
    STATUS_FAIL = STATUS_FAIL; WAIT = WAIT; HINT = HINT


I = _Icons()
