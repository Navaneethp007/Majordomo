"""The resident tray icon — Majordomo's presence on the machine.

Deliberately thin. It owns no logic of its own: it starts the panel, opens a
browser at it, and calls the same ``brief.run`` the CLI does. Everything
interesting lives one layer down and is testable without a GUI.

``pystray`` and ``Pillow`` are an optional extra, so ``pip install majordomo``
gets you a working CLI without dragging in a GUI toolkit. The import failure is
turned into an instruction rather than a traceback.
"""
from __future__ import annotations

import threading
import webbrowser

from majordomo import brief, panel, tts
from majordomo.config import Config


class TrayUnavailable(Exception):
    """The optional tray dependencies are not installed."""


def _load_pystray():
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise TrayUnavailable(
            "the tray needs extra packages — run: pip install majordomo[tray]"
        ) from exc
    return pystray, Image, ImageDraw


def _icon_image(Image, ImageDraw, size: int = 64):
    """A small bell, drawn rather than shipped so there's no binary asset."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fg = (232, 232, 232, 255)
    draw.pieslice([12, 14, 52, 54], start=180, end=360, fill=fg)
    draw.rectangle([12, 34, 52, 44], fill=fg)
    draw.rectangle([8, 44, 56, 49], fill=fg)
    draw.ellipse([27, 49, 37, 58], fill=fg)
    return image


def run(config: Config) -> None:
    """Run the tray icon. Blocks until quit."""
    pystray, Image, ImageDraw = _load_pystray()

    handle = panel.serve(config)

    def open_panel(icon=None, item=None) -> None:
        webbrowser.open(handle.url)

    def speak_now(icon=None, item=None) -> None:
        def work() -> None:
            try:
                result = brief.run(config)
                tts.speak(result.briefing.briefing_text, config.voice)
            except Exception:
                # A tray menu item must never take the resident process down.
                pass

        threading.Thread(target=work, daemon=True).start()

    def quit_now(icon, item) -> None:
        handle.stop()
        icon.stop()

    icon = pystray.Icon(
        "majordomo",
        _icon_image(Image, ImageDraw),
        "Majordomo",
        menu=pystray.Menu(
            pystray.MenuItem("Open briefing", open_panel, default=True),
            pystray.MenuItem("Speak briefing", speak_now),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", quit_now),
        ),
    )
    icon.run()
