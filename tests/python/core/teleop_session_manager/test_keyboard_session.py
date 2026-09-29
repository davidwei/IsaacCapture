# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A keyboard-only TeleopSession runs for real, with no OpenXR runtime and no mocks.

The in-process KeyboardTracker needs no OpenXR, so TeleopSession skips the OpenXR session.
Input comes from FakeKeyEventSource, which follows the KeyEventSource contract (focus loss
and host UI keyboard capture release held keys; press-only surfaces report taps).
"""

import numpy as np
import pytest

import isaaccapture.plugin_manager as pm
from isaaccapture import deviceio, oxr
from isaaccapture.deviceio_trackers import HeadTracker, KeyboardTracker
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    FakeKeyEventSource,
    KeyboardSource,
)
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

KEY_W, KEY_K = 17, 37


def _held(result):
    return np.flatnonzero(np.asarray(result["keyboard_held"][0])).tolist()


def _pressed(result):
    return np.flatnonzero(np.asarray(result["keyboard_pressed"][0])).tolist()


def test_requires_openxr_reflects_trackers():
    assert not deviceio.DeviceIOSession.requires_openxr([KeyboardTracker()])
    assert deviceio.DeviceIOSession.requires_openxr([KeyboardTracker(), HeadTracker()])


def test_keyboard_session_runs_without_openxr():
    keyboard = KeyboardSource(name="keyboard")
    surface = FakeKeyEventSource()
    detach = keyboard.attach(surface)
    config = TeleopSessionConfig(app_name="KeyboardNoOpenXR", pipeline=keyboard)

    with TeleopSession(config) as session:
        assert session.oxr_session is None

        surface.press("KeyW")
        result = session.step()
        assert _held(result) == [KEY_W]
        assert _pressed(result) == [KEY_W]

        surface.tap("KeyK")  # press-only surface: pressed this frame, never held
        result = session.step()
        assert _held(result) == [KEY_W]
        assert _pressed(result) == [KEY_K]

        surface.begin_text_input()  # UI takes the keyboard: held keys are released
        surface.press("KeyK")  # typed into the text field: not forwarded
        result = session.step()
        assert _held(result) == []
        assert _pressed(result) == []

        surface.end_text_input()
        surface.press("KeyW")
        surface.blur()  # focus loss releases it again
        result = session.step()
        assert _held(result) == []
        assert _pressed(result) == [KEY_W]

    detach()
    assert surface.listener_count == 0
    assert surface.capture_history == [True, False]


def test_keyboard_session_records_and_replays_without_openxr(tmp_path):
    from isaaccapture.deviceio_session import McapRecordingConfig, McapReplayConfig

    mcap_path = str(tmp_path / "keyboard.mcap")
    keyboard = KeyboardSource(name="keyboard")
    surface = FakeKeyEventSource()
    keyboard.attach(surface)

    live = []
    config = TeleopSessionConfig(
        app_name="KeyboardRecord",
        pipeline=keyboard,
        mcap_config=McapRecordingConfig(mcap_path),
    )
    with TeleopSession(config) as session:
        for action in (
            lambda: surface.press("KeyW"),
            lambda: surface.tap("KeyK"),
            surface.blur,
        ):
            action()
            result = session.step()
            live.append((_held(result), _pressed(result)))

    replay_config = TeleopSessionConfig(
        app_name="KeyboardReplay",
        pipeline=KeyboardSource(name="keyboard"),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(mcap_path),
    )
    replayed = []
    with TeleopSession(replay_config) as session:
        for _ in live:
            result = session.step()
            replayed.append((_held(result), _pressed(result)))

    assert live == [([KEY_W], [KEY_W]), ([KEY_W], [KEY_K]), ([], [])]
    assert replayed == live


class _OpenXRConstructed(Exception):
    pass


def _forbid_openxr(monkeypatch):
    def refuse(*args, **kwargs):
        raise _OpenXRConstructed

    monkeypatch.setattr(oxr, "OpenXRSession", refuse)


def _plugin(name, tmp_path, *, enabled, required):
    return PluginConfig(
        plugin_name=name,
        plugin_root_id=f"/{name}",
        search_paths=[tmp_path / "missing"],
        enabled=enabled,
        required=required,
    )


def test_disabled_plugins_do_not_require_openxr(monkeypatch, tmp_path):
    _forbid_openxr(monkeypatch)
    config = TeleopSessionConfig(
        app_name="KeyboardDisabledPlugins",
        pipeline=KeyboardSource(name="keyboard"),
        plugins=[
            _plugin("a", tmp_path, enabled=False, required=True),
            _plugin("b", tmp_path, enabled=False, required=True),
        ],
    )

    with TeleopSession(config) as session:
        assert session.oxr_session is None
        assert session.plugin_contexts == []


def test_unavailable_optional_plugin_does_not_require_openxr(monkeypatch, tmp_path):
    """An enabled plugin that cannot be found is skipped, so it must not force OpenXR."""
    _forbid_openxr(monkeypatch)
    config = TeleopSessionConfig(
        app_name="KeyboardMissingOptionalPlugin",
        pipeline=KeyboardSource(name="keyboard"),
        plugins=[_plugin("b", tmp_path, enabled=True, required=False)],
    )

    with TeleopSession(config) as session:
        assert session.oxr_session is None
        assert session.plugin_contexts == []


def test_unavailable_required_plugin_fails_before_openxr(monkeypatch, tmp_path):
    _forbid_openxr(monkeypatch)
    config = TeleopSessionConfig(
        app_name="KeyboardMissingRequiredPlugin",
        pipeline=KeyboardSource(name="keyboard"),
        plugins=[_plugin("b", tmp_path, enabled=True, required=True)],
    )

    with pytest.raises(RuntimeError, match="Required plugin 'b'"):
        with TeleopSession(config):
            pass


def test_launchable_plugin_requires_openxr(monkeypatch, tmp_path):
    class _Manager:
        def __init__(self, search_paths):
            pass

        def get_plugin_names(self):
            return ["b"]

    _forbid_openxr(monkeypatch)
    monkeypatch.setattr(pm, "PluginManager", _Manager)
    config = TeleopSessionConfig(
        app_name="KeyboardLaunchablePlugin",
        pipeline=KeyboardSource(name="keyboard"),
        plugins=[
            PluginConfig(
                plugin_name="b",
                plugin_root_id="/b",
                search_paths=[tmp_path],
                required=True,
            )
        ],
    )

    with pytest.raises(_OpenXRConstructed):
        with TeleopSession(config):
            pass
