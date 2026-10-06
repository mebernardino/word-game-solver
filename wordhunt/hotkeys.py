"""Global hotkeys via pynput (work even while the game window has focus)."""
from __future__ import annotations

from typing import Callable, Dict


def key_name(key) -> str:
    """Normalize a pynput key to a lowercase name like 'esc', 'f8' or 'a'."""
    name = getattr(key, "name", None)  # special keys (Key.esc -> 'esc')
    if name:
        return name.lower()
    char = getattr(key, "char", None)
    return char.lower() if char else ""


class Hotkeys:
    """Calls bindings[name]() from a background thread when that key is pressed."""

    def __init__(self, bindings: Dict[str, Callable[[], None]]):
        self.bindings = {k.lower(): v for k, v in bindings.items()}
        self._listener = None

    def start(self) -> "Hotkeys":
        from pynput import keyboard

        def on_press(key):
            fn = self.bindings.get(key_name(key))
            if fn:
                fn()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()
        return self

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def __enter__(self) -> "Hotkeys":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
