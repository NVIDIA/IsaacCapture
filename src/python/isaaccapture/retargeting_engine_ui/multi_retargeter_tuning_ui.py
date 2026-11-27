# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-retargeter ImGui tuning UI - unified window for multiple retargeters.

Provides an async ImGui window that displays tuning controls for multiple
retargeters in a single window with configurable layouts.
"""

import logging
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import glfw  # type: ignore[import-not-found]
import imgui  # type: ignore[import-not-found]
from imgui.integrations.glfw import GlfwRenderer  # type: ignore[import-not-found]

from isaaccapture.retargeting_engine.interface import (
    BoolParameter,
    FloatParameter,
    IntParameter,
    ParameterSpec,
    VectorParameter,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaccapture.retargeting_engine.interface import BaseRetargeter, ParameterState


# ============================================================================
# ImGui Helper Functions
# ============================================================================


def render_parameter_ui(
    param_name: str, param_spec: ParameterSpec, current_value: Any
) -> Tuple[bool, Any]:
    """
    Render ImGui controls for a single parameter.

    Args:
        param_name: Name of the parameter
        param_spec: Parameter specification
        current_value: Current value of the parameter (from snapshot)

    Returns:
        Tuple of (changed, new_value) - changed is True if value was modified
    """
    changed = False
    new_value = current_value

    if isinstance(param_spec, BoolParameter):
        changed, new_value = imgui.checkbox(
            f"{param_name}: {param_spec.description}", current_value
        )

    elif isinstance(param_spec, FloatParameter):
        imgui.text(f"{param_name}:")
        imgui.text_colored(param_spec.description, 0.6, 0.6, 0.6)
        imgui.push_item_width(-1)
        changed, new_value = imgui.slider_float(
            f"##{param_name}",
            current_value,
            param_spec.min_value,
            param_spec.max_value,
            format="%.3f",
        )
        imgui.pop_item_width()

    elif isinstance(param_spec, IntParameter):
        imgui.text(f"{param_name}:")
        imgui.text_colored(param_spec.description, 0.6, 0.6, 0.6)
        imgui.push_item_width(-1)
        changed, new_value = imgui.slider_int(
            f"##{param_name}", current_value, param_spec.min_value, param_spec.max_value
        )
        imgui.pop_item_width()

    elif isinstance(param_spec, VectorParameter):
        imgui.text(f"{param_name}:")
        imgui.text_colored(param_spec.description, 0.6, 0.6, 0.6)

        assert param_spec.element_names is not None, (
            "VectorParameter must have element_names"
        )
        current_vector = list(current_value)
        vector_changed = False

        for i, elem_name in enumerate(param_spec.element_names):
            elem_min = (
                float(param_spec.min_value[i])
                if param_spec.min_value is not None
                else 0.0
            )  # type: ignore[index]
            elem_max = (
                float(param_spec.max_value[i])
                if param_spec.max_value is not None
                else 1.0
            )  # type: ignore[index]

            imgui.text(f"  {elem_name}:")
            imgui.same_line()
            imgui.push_item_width(-100)
            elem_changed, new_elem_value = imgui.slider_float(
                f"##{param_name}_{i}",
                current_vector[i],
                elem_min,
                elem_max,
                format="%.3f",
            )
            imgui.pop_item_width()

            if elem_changed:
                current_vector[i] = new_elem_value
                vector_changed = True

        if vector_changed:
            changed = True
            new_value = current_vector

    return changed, new_value


# ============================================================================
# Layout Modes
# ============================================================================


class LayoutModeImGui(Enum):
    """Layout modes for multi-retargeter ImGui UI."""

    TABS = "tabs"  # Tabbed interface (one tab per retargeter)
    VERTICAL = "vertical"  # Vertical stack (one below another)
    HORIZONTAL = "horizontal"  # Horizontal layout (side by side)
    FLOATING = "floating"  # Free-floating separate ImGui windows


class MultiRetargeterTuningUIImGui:
    """
    Async ImGui tuning UI for multiple retargeters in a single window.

    Runs automatically in a background thread. Supports RAII via context managers.

    Example:
        retargeters = [processor_a, processor_b, processor_c]

        # RAII style (recommended):
        with MultiRetargeterTuningUIImGui(retargeters, layout_mode=LayoutModeImGui.TABS) as ui:
            while running:
                # Process data - UI updates automatically in background
                processor_a(inputs, outputs)

        # Manual style:
        ui = MultiRetargeterTuningUIImGui(retargeters)
        ui.start()
        # ... use retargeters ...
        ui.stop()
    """

    def __init__(
        self,
        retargeters: List["BaseRetargeter"],
        title: str = "Multi-Retargeter Tuning",
        layout_mode: LayoutModeImGui = LayoutModeImGui.TABS,
    ):
        """
        Initialize and start multi-retargeter tuning UI (RAII).

        The UI starts automatically in a background thread and stops when
        the object is destroyed or when used as a context manager.

        Args:
            retargeters: List of retargeter instances to tune
            title: Window title
            layout_mode: How to organize retargeters in the window

        Raises:
            ImportError: If imgui/glfw is not installed
            ValueError: If no retargeters provided or none have parameters

        Example:
            # RAII with context manager (recommended):
            with MultiRetargeterTuningUIImGui([proc_a, proc_b]) as ui:
                while running:
                    # Process data
                    pass
            # UI automatically stops here

            # Or direct construction (stops in destructor):
            ui = MultiRetargeterTuningUIImGui([proc_a, proc_b])
            # ... use retargeters ...
            # UI stops when ui goes out of scope
        """

        if not retargeters:
            raise ValueError("At least one retargeter must be provided")

        # Extract and store only ParameterState objects (thread-safe)
        self._parameter_states: Dict[str, "ParameterState"] = {}
        for ret in retargeters:
            param_state = (
                ret.get_parameter_state()
                if hasattr(ret, "get_parameter_state")
                else None
            )
            if (
                param_state is not None
                and len(param_state.get_all_parameter_specs()) > 0
            ):
                name = ret._name if hasattr(ret, "_name") else str(ret)
                self._parameter_states[name] = param_state

        if not self._parameter_states:
            raise ValueError("No retargeters with tunable parameters found")

        self._title = title
        self._layout_mode = layout_mode
        self._window = None
        self._impl = None
        self._is_open = False

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        # Status message
        self._status_message = ""
        self._status_success = True
        self._status_time = 0.0

        # Tab state
        self._current_tab = 0

        # Floating window initial positioning flag
        self._floating_windows_positioned = False
        self._floating_auto_arrange = False  # Flag to trigger auto-arrange

        # Start UI automatically (RAII)
        self._start()

    def _render_retargeter_params(
        self, name: str, param_state: "ParameterState"
    ) -> None:
        """Render parameter controls for a single retargeter."""
        parameters = param_state.get_all_parameter_specs()
        if not parameters:
            imgui.text("No tunable parameters")
            return

        # Get snapshot of all values
        values = param_state.get_all_values()

        # Collect all changes
        updates = {}
        for param_name, param_spec in parameters.items():
            changed, new_value = render_parameter_ui(
                param_name, param_spec, values[param_name]
            )
            if changed:
                updates[param_name] = new_value
            imgui.separator()

        # Apply all changes in one batch
        if updates:
            param_state.set(updates)

        # Individual action buttons
        imgui.spacing()

        if imgui.button(f"Save##{name}", width=100):
            if param_state.save_to_file():
                self._set_status(f"✓ {name}: Configuration saved!", success=True)
            else:
                self._set_status(f"✗ {name}: Failed to save", success=False)

        imgui.same_line()

        if imgui.button(f"Reset to Saved##{name}", width=130):
            if param_state.load_from_file():
                self._set_status(f"✓ {name}: Reset to saved values", success=True)
            else:
                self._set_status(f"✗ {name}: No saved config", success=False)

        imgui.same_line()

        if imgui.button(f"Reset to Defaults##{name}", width=150):
            param_state.reset_to_defaults()
            self._set_status(f"✓ {name}: Reset to defaults", success=True)

    def _render_tabbed_layout(self) -> None:
        """Render tabbed layout with one tab per retargeter."""
        if imgui.begin_tab_bar("RetargeterTabs"):
            for name, param_state in self._parameter_states.items():
                if imgui.begin_tab_item(name)[0]:
                    imgui.begin_child(f"tab_content_{name}", border=False)
                    self._render_retargeter_params(name, param_state)
                    imgui.end_child()
                    imgui.end_tab_item()
            imgui.end_tab_bar()

    def _render_vertical_layout(self) -> None:
        """Render vertical stack layout."""
        for name, param_state in self._parameter_states.items():
            if imgui.collapsing_header(name, flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
                imgui.indent()
                self._render_retargeter_params(name, param_state)
                imgui.unindent()
                imgui.spacing()

    def _render_horizontal_layout(self) -> None:
        """Render horizontal layout."""
        num_retargeters = len(self._parameter_states)

        imgui.columns(num_retargeters, "retargeter_columns")

        for idx, (name, param_state) in enumerate(self._parameter_states.items()):
            if idx > 0:
                imgui.next_column()

            imgui.text(name)
            imgui.separator()
            imgui.begin_child(f"col_{name}", height=-1, border=True)
            self._render_retargeter_params(name, param_state)
            imgui.end_child()

        imgui.columns(1)

    def _render_floating_layout(self) -> None:
        """Render free-floating ImGui windows for each retargeter."""
        window_width, window_height = glfw.get_window_size(self._window)
        num_windows = len(self._parameter_states)

        # Calculate grid layout for snap-to-fit
        if self._floating_auto_arrange:
            # Determine grid dimensions
            cols = int(np.ceil(np.sqrt(num_windows)))
            rows = int(np.ceil(num_windows / cols))

            # Calculate window size to fit in grid
            margin = 10
            controls_height = 140  # Reserve space for global controls
            available_width = window_width - margin * (cols + 1)
            available_height = window_height - controls_height - margin * (rows + 1)

            win_width = available_width // cols
            win_height = available_height // rows

        # Each retargeter gets its own movable/resizable ImGui window
        for idx, (name, param_state) in enumerate(self._parameter_states.items()):
            # Position windows
            if self._floating_auto_arrange:
                # Grid layout
                col = idx % cols
                row = idx // cols
                x = margin + col * (win_width + margin)
                y = controls_height + margin + row * (win_height + margin)
                imgui.set_next_window_position(x, y)
                imgui.set_next_window_size(win_width, win_height)
            elif not self._floating_windows_positioned:
                # Cascade on first render
                imgui.set_next_window_position(
                    100 + idx * 30, 100 + idx * 30, imgui.FIRST_USE_EVER
                )
                imgui.set_next_window_size(400, 500, imgui.FIRST_USE_EVER)

            # Create independent floating window
            expanded, _opened = imgui.begin(f"{name}##floating", closable=False)
            if expanded:
                self._render_retargeter_params(name, param_state)
            imgui.end()

        # Mark as positioned after first frame
        self._floating_windows_positioned = True
        # Clear auto-arrange flag after applying
        self._floating_auto_arrange = False

    def _render_ui(self) -> None:
        """Render the main UI."""
        # For floating mode, skip the main container window and just render floating windows
        if self._layout_mode == LayoutModeImGui.FLOATING:
            self._render_floating_layout()

            # Render global control bar as a separate floating window
            imgui.set_next_window_position(10, 10, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(650, 120, imgui.FIRST_USE_EVER)
            imgui.begin("Global Controls##floating", closable=False)

            imgui.text(f"{self._title} - Global Controls")
            imgui.separator()
            imgui.spacing()

            if imgui.button("Save All", width=100):
                self._save_all()

            imgui.same_line()

            if imgui.button("Reset to Saved", width=120):
                for state in self._parameter_states.values():
                    state.load_from_file()
                self._set_status("✓ All parameters reset to saved values", success=True)

            imgui.same_line()

            if imgui.button("Reset to Defaults", width=140):
                for state in self._parameter_states.values():
                    state.reset_to_defaults()
                self._set_status(
                    "✓ All parameters reset to default values", success=True
                )

            imgui.same_line()

            if imgui.button("Snap to Fit", width=100):
                self._floating_auto_arrange = True
                self._set_status("✓ Windows arranged in grid", success=True)

            # Status bar
            if self._status_message and (time.time() - self._status_time) < 3.0:
                imgui.spacing()
                imgui.separator()
                color = (0.3, 0.8, 0.3) if self._status_success else (0.8, 0.3, 0.3)
                imgui.text_colored(self._status_message, *color)

            imgui.end()
            return

        # For other modes, use a main container window
        # Main window
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(*glfw.get_window_size(self._window))

        imgui.begin(
            self._title,
            closable=False,
            flags=imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE,
        )

        # Render layout based on mode
        content_height = (
            imgui.get_window_height() - 150
        )  # Reserve space for buttons and status

        imgui.begin_child("parameters", height=content_height)

        if self._layout_mode == LayoutModeImGui.TABS:
            self._render_tabbed_layout()
        elif self._layout_mode == LayoutModeImGui.VERTICAL:
            self._render_vertical_layout()
        elif self._layout_mode == LayoutModeImGui.HORIZONTAL:
            self._render_horizontal_layout()

        imgui.end_child()

        # Global action buttons
        imgui.separator()
        imgui.spacing()

        if imgui.button("Save All", width=120):
            self._save_all()

        imgui.same_line()

        if imgui.button("Reset All to Saved", width=150):
            for state in self._parameter_states.values():
                state.load_from_file()
            self._set_status("✓ All parameters reset to saved values", success=True)

        imgui.same_line()

        if imgui.button("Reset All to Defaults", width=170):
            for state in self._parameter_states.values():
                state.reset_to_defaults()
            self._set_status("✓ All parameters reset to default values", success=True)

        # Status bar
        if self._status_message and (time.time() - self._status_time) < 3.0:
            imgui.spacing()
            imgui.separator()
            color = (0.3, 0.8, 0.3) if self._status_success else (0.8, 0.3, 0.3)
            imgui.text_colored(self._status_message, *color)

        imgui.end()

    def _open(self) -> None:
        """Open the ImGui window (called on UI thread)."""
        if self._is_open:
            return

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Create window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        window_width = 900 if self._layout_mode == LayoutModeImGui.HORIZONTAL else 700
        if self._layout_mode == LayoutModeImGui.FLOATING:
            window_width = 1280
            window_height = 960
        else:
            window_height = 800

        self._window = glfw.create_window(
            window_width, window_height, self._title, None, None
        )
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1)  # Enable vsync

        # Initialize ImGui
        imgui.create_context()
        self._impl = GlfwRenderer(self._window)

        # Style
        style = imgui.get_style()
        style.window_rounding = 5.0
        style.frame_rounding = 3.0
        style.scrollbar_rounding = 3.0

        # Colors - dark theme
        imgui.style_colors_dark()

        self._is_open = True

    def _update(self) -> bool:
        """Update UI (called on UI thread)."""
        if not self._is_open or self._window is None:
            return False

        # Check if window should close
        if glfw.window_should_close(self._window):
            return False

        # Poll events
        glfw.poll_events()
        self._impl.process_inputs()

        # Clear the framebuffer to prevent shadow artifacts
        import OpenGL.GL as gl  # type: ignore[import-not-found]

        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Start new frame
        imgui.new_frame()

        # Render UI
        self._render_ui()

        # Render
        imgui.render()
        glfw.make_context_current(self._window)
        imgui_draw_data = imgui.get_draw_data()
        if imgui_draw_data:
            self._impl.render(imgui_draw_data)
        glfw.swap_buffers(self._window)

        return True

    def _close(self) -> None:
        """Close the ImGui window (called on UI thread)."""
        if self._impl is not None:
            self._impl.shutdown()
            self._impl = None

        if self._window is not None:
            glfw.destroy_window(self._window)
            self._window = None

        glfw.terminate()
        self._is_open = False

    def _save_all(self) -> None:
        failed = []
        # Attempt every save even when another state has no path or cannot be written.
        for name, state in self._parameter_states.items():
            if not state.save_to_file():
                failed.append(name)
        if failed:
            self._set_status(f"✗ Failed to save: {', '.join(failed)}", success=False)
        else:
            self._set_status("✓ All configurations saved successfully!", success=True)

    def _set_status(self, message: str, success: bool = True) -> None:
        """Set status message."""
        self._status_message = message
        self._status_success = success
        self._status_time = time.time()

    def _start(self) -> None:
        """Start the UI thread (called automatically in __init__)."""
        if self._running:
            return

        try:
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._run_ui_loop, daemon=True)
            self._thread.start()

        except Exception as e:
            logger.error("Failed to start: %s", e)
            self._running = False

    def _run_ui_loop(self):
        """UI thread main loop."""
        try:
            # Open the window
            self._open()

            # Event loop
            while self._running and not self._stop_event.is_set():
                if not self._update():
                    break

                # vsync is enabled, but add a small sleep as backup
                # to cap at ~120Hz if vsync is not working properly
                time.sleep(1.0 / 120.0)

        finally:
            self._close()
            self._running = False

    def _stop(self) -> None:
        """Stop the async UI and wait for thread to finish."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._thread = None

    def __del__(self):
        """Destructor - stops UI automatically (RAII)."""
        self._stop()

    def is_running(self) -> bool:
        """Check if the async UI is currently running."""
        return self._running

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops UI."""
        self._stop()
        return False
