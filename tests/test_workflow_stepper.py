"""Tests for the WorkflowStepper UX widget (Option-D redesign).

Covers:
  * StepStatus transitions and header-bar paint helpers
  * WorkflowStepper — initial state, navigation, lock/unlock, mark_done
  * Auto-advance (mark_done_and_advance)
  * step_changed signal
  * Navigation buttons enabled/disabled state
  * Keyboard/menu step navigation (go_to)
"""
from __future__ import annotations

import pytest
from PyQt5.QtWidgets import QLabel, QWidget

from prospective.ui.widgets.workflow_stepper import (
    WorkflowStepper,
    WorkflowStepDef,
    StepStatus,
    _StepHeaderBar,
    _HelpBanner,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def _make_panel(name: str = "") -> QWidget:
    lbl = QLabel(name or "panel")
    return lbl


def _make_steps(n: int = 6) -> list[WorkflowStepDef]:
    titles = ["Paciente", "Segmentación", "Detección",
              "Morfometría", "Planificación", "Exportar"]
    icons  = ["📂", "🔬", "🎯", "📊", "🩺", "📄"]
    descs  = [f"Descripción del paso {i+1}." for i in range(n)]
    return [
        WorkflowStepDef(
            key=f"step{i}",
            number=i + 1,
            icon=icons[i % len(icons)],
            title=titles[i % len(titles)],
            description=descs[i],
            panel=_make_panel(titles[i % len(titles)]),
        )
        for i in range(n)
    ]


@pytest.fixture()
def stepper(qapp) -> WorkflowStepper:
    steps = _make_steps(6)
    return WorkflowStepper(steps)


@pytest.fixture()
def small_stepper(qapp) -> WorkflowStepper:
    """3-step stepper with all steps unlocked for convenience."""
    steps = _make_steps(3)
    s = WorkflowStepper(steps)
    s.unlock(1)
    s.unlock(2)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# StepStatus
# ══════════════════════════════════════════════════════════════════════════════

class TestStepStatus:

    def test_enum_members_exist(self):
        assert StepStatus.LOCKED
        assert StepStatus.AVAILABLE
        assert StepStatus.ACTIVE
        assert StepStatus.DONE

    def test_all_different(self):
        statuses = [StepStatus.LOCKED, StepStatus.AVAILABLE,
                    StepStatus.ACTIVE, StepStatus.DONE]
        assert len(set(statuses)) == 4


# ══════════════════════════════════════════════════════════════════════════════
# _StepHeaderBar
# ══════════════════════════════════════════════════════════════════════════════

class TestStepHeaderBar:

    def test_initial_status_locked_except_first(self, qapp):
        steps = _make_steps(4)
        bar = _StepHeaderBar(steps)
        assert bar.get_status(0) == StepStatus.AVAILABLE
        for i in range(1, 4):
            assert bar.get_status(i) == StepStatus.LOCKED

    def test_set_status_changes_status(self, qapp):
        bar = _StepHeaderBar(_make_steps(3))
        bar.set_status(1, StepStatus.ACTIVE)
        assert bar.get_status(1) == StepStatus.ACTIVE

    def test_set_status_all_variants(self, qapp):
        bar = _StepHeaderBar(_make_steps(4))
        for idx, status in enumerate([StepStatus.DONE, StepStatus.ACTIVE,
                                       StepStatus.AVAILABLE, StepStatus.LOCKED]):
            bar.set_status(idx, status)
            assert bar.get_status(idx) == status

    def test_clicked_signal_emitted_for_available(self, qapp, qtbot):
        bar = _StepHeaderBar(_make_steps(3))
        bar.set_status(1, StepStatus.AVAILABLE)
        with qtbot.waitSignal(bar.clicked, timeout=500) as blocker:
            # Simulate click at the approximate x-position of step 1
            # (steps are evenly spaced; bar width unknown here, use resize)
            bar.resize(300, 62)
            from PyQt5.QtCore import QPoint
            from PyQt5.QtTest import QTest
            from PyQt5.QtCore import Qt
            mid_x = int(300 * 1.5 / 3)   # centre of step index 1
            QTest.mouseClick(bar, Qt.LeftButton, pos=QPoint(mid_x, 31))
        assert blocker.args[0] == 1

    def test_clicked_not_emitted_for_locked(self, qapp, qtbot):
        bar = _StepHeaderBar(_make_steps(3))
        bar.resize(300, 62)
        from PyQt5.QtCore import QPoint
        from PyQt5.QtTest import QTest
        from PyQt5.QtCore import Qt
        received = []
        bar.clicked.connect(received.append)
        mid_x = int(300 * 1.5 / 3)
        QTest.mouseClick(bar, Qt.LeftButton, pos=QPoint(mid_x, 31))
        assert received == []


# ══════════════════════════════════════════════════════════════════════════════
# _HelpBanner
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpBanner:

    def test_set_content_updates_labels(self, qapp):
        banner = _HelpBanner()
        banner.set_content("Paso 3 — 🎯 Detección",
                           "Selecciona un candidato.")
        assert "Detección" in banner._title.text()
        assert "Selecciona" in banner._desc.text()

    def test_toggle_hides_description(self, qapp):
        banner = _HelpBanner()
        banner.set_content("Título", "Descripción larga.")
        # Use isHidden() — isVisible() requires the parent to be shown too
        assert not banner._desc.isHidden()
        banner._toggle()
        assert banner._desc.isHidden()
        banner._toggle()
        assert not banner._desc.isHidden()


# ══════════════════════════════════════════════════════════════════════════════
# WorkflowStepper — initial state
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowStepperInitial:

    def test_current_index_is_zero(self, stepper):
        assert stepper.current_index() == 0

    def test_step0_is_active(self, stepper):
        assert stepper._header.get_status(0) == StepStatus.ACTIVE

    def test_steps_1_to_5_locked(self, stepper):
        for i in range(1, 6):
            assert stepper._header.get_status(i) == StepStatus.LOCKED

    def test_stack_shows_first_panel(self, stepper):
        assert stepper._stack.currentIndex() == 0

    def test_prev_button_disabled_at_start(self, stepper):
        assert not stepper._btn_prev.isEnabled()

    def test_next_button_disabled_when_all_locked(self, stepper):
        # All steps except 0 are locked → next disabled
        assert not stepper._btn_next.isEnabled()

    def test_step_keys(self, stepper):
        assert stepper.step_key(0) == "step0"
        assert stepper.step_key(5) == "step5"

    def test_help_banner_shows_step1_content(self, stepper):
        assert "Paciente" in stepper._help._title.text()


# ══════════════════════════════════════════════════════════════════════════════
# WorkflowStepper — unlock / mark_done
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowStepperLocking:

    def test_unlock_changes_locked_to_available(self, stepper):
        stepper.unlock(1)
        assert stepper._header.get_status(1) == StepStatus.AVAILABLE

    def test_unlock_does_not_downgrade_done(self, stepper):
        stepper.mark_done(0)
        stepper.unlock(0)   # should stay DONE, not regress to AVAILABLE
        assert stepper._header.get_status(0) == StepStatus.DONE

    def test_unlock_enables_next_button(self, stepper):
        stepper.unlock(1)
        assert stepper._btn_next.isEnabled()

    def test_mark_done_sets_done_status(self, stepper):
        stepper.mark_done(0)
        assert stepper._header.get_status(0) == StepStatus.DONE

    def test_mark_done_and_advance_moves_to_next(self, stepper):
        stepper.mark_done_and_advance(0)
        assert stepper.current_index() == 1
        assert stepper._header.get_status(0) == StepStatus.DONE
        assert stepper._header.get_status(1) == StepStatus.ACTIVE

    def test_mark_done_and_advance_at_last_step_stays(self, stepper):
        # Unlock all first
        for i in range(6):
            stepper.unlock(i)
        stepper.go_to(5)
        stepper.mark_done_and_advance(5)   # no step 6
        assert stepper.current_index() == 5   # stays at last


# ══════════════════════════════════════════════════════════════════════════════
# WorkflowStepper — navigation
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowStepperNavigation:

    def test_go_to_available_step(self, small_stepper):
        small_stepper.go_to(1)
        assert small_stepper.current_index() == 1

    def test_go_to_locked_step_does_nothing(self, stepper):
        stepper.go_to(3)   # step 3 is LOCKED
        assert stepper.current_index() == 0

    def test_go_to_updates_stack(self, small_stepper):
        small_stepper.go_to(2)
        assert small_stepper._stack.currentIndex() == 2

    def test_go_to_updates_help_banner(self, small_stepper):
        small_stepper.go_to(1)
        assert "Segmentación" in small_stepper._help._title.text()

    def test_go_to_prev_enabled_after_advance(self, small_stepper):
        small_stepper.go_to(1)
        assert small_stepper._btn_prev.isEnabled()

    def test_go_to_prev_disabled_at_first_step(self, small_stepper):
        small_stepper.go_to(0)
        assert not small_stepper._btn_prev.isEnabled()

    def test_step_changed_signal_on_advance(self, small_stepper, qtbot):
        with qtbot.waitSignal(small_stepper.step_changed, timeout=500) as blocker:
            small_stepper.go_to(1)
        assert blocker.args[0] == 1

    def test_step_changed_not_emitted_when_same_step(self, small_stepper, qtbot):
        received = []
        small_stepper.step_changed.connect(received.append)
        small_stepper.go_to(0)   # already at 0
        assert received == []

    def test_panel_method_returns_correct_widget(self, stepper):
        steps = _make_steps(3)
        s = WorkflowStepper(steps)
        assert s.panel(0) is steps[0].panel
        assert s.panel(2) is steps[2].panel

    def test_prev_button_navigates_back(self, small_stepper, qtbot):
        small_stepper.go_to(2)
        small_stepper._btn_prev.click()
        assert small_stepper.current_index() == 1

    def test_next_button_navigates_forward(self, small_stepper, qtbot):
        small_stepper._btn_next.click()
        assert small_stepper.current_index() == 1

    def test_next_button_text_includes_next_title(self, small_stepper):
        # At step 0 the next button should mention step 1's title ("Segmentación")
        text = small_stepper._btn_next.text()
        assert "Siguiente" in text
        assert "Segmentaci" in text   # avoids encoding mismatch on Windows

    def test_header_bar_click_navigates(self, small_stepper, qtbot):
        # Emit the header's clicked signal directly (avoids pixel-level test)
        small_stepper._header.clicked.emit(2)
        assert small_stepper.current_index() == 2


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkflowStepperEdgeCases:

    def test_empty_stepper_no_crash(self, qapp):
        s = WorkflowStepper([])
        assert s.current_index() == 0

    def test_single_step_no_crash(self, qapp):
        steps = _make_steps(1)
        s = WorkflowStepper(steps)
        assert s.current_index() == 0
        assert not s._btn_next.isEnabled()
        assert not s._btn_prev.isEnabled()

    def test_go_to_out_of_range_does_nothing(self, small_stepper):
        small_stepper.go_to(99)
        assert small_stepper.current_index() == 0
        small_stepper.go_to(-1)
        assert small_stepper.current_index() == 0

    def test_unlock_idempotent(self, stepper):
        stepper.unlock(1)
        stepper.unlock(1)   # second call — no error
        assert stepper._header.get_status(1) == StepStatus.AVAILABLE

    def test_mark_done_idempotent(self, stepper):
        stepper.mark_done(0)
        stepper.mark_done(0)
        assert stepper._header.get_status(0) == StepStatus.DONE
