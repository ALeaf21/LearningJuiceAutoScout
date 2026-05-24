import builtins
import os
import sys


ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "info": "\033[1;36m",
    "warn": "\033[1;33m",
    "error": "\033[1;31m",
    "done": "\033[1;32m",
    "path": "\033[0;94m",
    "shot_made": "\033[1;32m",
    "shot_missed": "\033[1;31m",
}
COLOR_OUTPUT_ENABLED = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
ORIGINAL_PRINT = builtins.print
_ACTIVE_PROGRESS_BAR = None
_CONSOLE_INSTALLED = False


def _ansi(style: str, text: str) -> str:
    if not COLOR_OUTPUT_ENABLED:
        return text
    code = ANSI_STYLES.get(style)
    if not code:
        return text
    return "{}{}{}".format(code, text, ANSI_RESET)


def _style_console_text(text: str) -> str:
    if not COLOR_OUTPUT_ENABLED or not isinstance(text, str):
        return text

    stripped = text.lstrip("\r\n")
    prefix = text[:len(text) - len(stripped)]

    tag_styles = {
        "[INFO]": "info",
        "[WARN]": "warn",
        "[ERROR]": "error",
        "[DONE]": "done",
    }
    for tag, style in tag_styles.items():
        if stripped.startswith(tag):
            stripped = _ansi(style, tag) + stripped[len(tag):]
            break

    if "Shot made by" in stripped:
        stripped = stripped.replace("Shot made by", "{} made by".format(
            _ansi("shot_made", "Shot")), 1)
    elif "Shot missed by" in stripped:
        stripped = stripped.replace("Shot missed by", "{} missed by".format(
            _ansi("shot_missed", "Shot")), 1)

    if stripped.startswith("  CSV:"):
        stripped = "  CSV:    {}".format(_ansi("path", stripped.split("CSV:", 1)[1].strip()))
    elif stripped.startswith("  JLOG:"):
        stripped = "  JLOG:   {}".format(_ansi("path", stripped.split("JLOG:", 1)[1].strip()))
    elif stripped.startswith("  WPILOG:"):
        stripped = "  WPILOG: {}".format(_ansi("path", stripped.split("WPILOG:", 1)[1].strip()))
    elif stripped.startswith("  Debug:"):
        stripped = "  Debug:  {}".format(_ansi("path", stripped.split("Debug:", 1)[1].strip()))

    return prefix + stripped


def _clear_active_progress_bar():
    bar = _ACTIVE_PROGRESS_BAR
    if bar is None:
        return
    file = getattr(bar, "file", None)
    is_tty = getattr(bar, "is_tty", None)
    if file is None or not callable(is_tty) or not is_tty():
        return
    width = max(0, int(getattr(bar, "_max_width", 0)))
    if width > 0:
        ORIGINAL_PRINT("\r{}\r".format(" " * width), end="", file=file, flush=True)
    else:
        ORIGINAL_PRINT("\r", end="", file=file, flush=True)


def _redraw_active_progress_bar():
    bar = _ACTIVE_PROGRESS_BAR
    if bar is None:
        return
    update = getattr(bar, "update", None)
    if callable(update):
        update()


def console_print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", None)
    flush = kwargs.pop("flush", False)
    if kwargs:
        raise TypeError("Unsupported print kwargs: {}".format(", ".join(sorted(kwargs.keys()))))
    if file is None:
        file = sys.stdout
    text = sep.join(str(arg) for arg in args)
    if file in (sys.stdout, sys.stderr):
        text = _style_console_text(text)
    if _ACTIVE_PROGRESS_BAR is not None and file in (sys.stdout, sys.stderr):
        _clear_active_progress_bar()
    ORIGINAL_PRINT(text, end=end, file=file, flush=flush)
    if _ACTIVE_PROGRESS_BAR is not None and file in (sys.stdout, sys.stderr):
        _redraw_active_progress_bar()


def install_console_output() -> None:
    global _CONSOLE_INSTALLED
    if _CONSOLE_INSTALLED:
        return
    builtins.print = console_print
    _CONSOLE_INSTALLED = True


class _ManagedBar:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def next(self, n=1):
        return self._inner.next(n)

    def update(self):
        return self._inner.update()

    def finish(self):
        global _ACTIVE_PROGRESS_BAR
        try:
            return self._inner.finish()
        finally:
            if _ACTIVE_PROGRESS_BAR is self:
                _ACTIVE_PROGRESS_BAR = None


def _require(package, pip_name=None):
    import importlib

    try:
        return importlib.import_module(package)
    except ImportError:
        name = pip_name or package
        print("[ERROR] Missing: {}\n  Install: python3 -m pip install {}".format(name, name))
        sys.exit(1)


def _make_bar(label, max_val):
    global _ACTIVE_PROGRESS_BAR
    try:
        from progress.bar import Bar

        bar = _ManagedBar(Bar(
            label,
            max=max_val,
            suffix="%(percent).0f%% %(elapsed_td)s ETA %(eta_td)s",
        ))
        _ACTIVE_PROGRESS_BAR = bar
        return bar
    except ImportError:
        class _FallbackBar:
            def __init__(self, lbl, total):
                self._lbl = lbl
                self._total = max(total, 1)
                self._n = 0
                self.file = sys.stdout
                self._max_width = 0
                self._render()

            def is_tty(self):
                return self.file.isatty()

            def _render(self):
                pct = int(self._n / self._total * 100)
                line = "[{}] {}%".format(self._lbl, pct)
                self._max_width = max(self._max_width, len(line))
                ORIGINAL_PRINT("\r{}".format(line), end="", file=self.file, flush=True)

            def next(self, n=1):
                self._n += n
                if self._n % max(1, self._total // 20) == 0 or self._n >= self._total:
                    self._render()

            def update(self):
                self._render()

            def finish(self):
                ORIGINAL_PRINT(file=self.file, flush=True)

        bar = _ManagedBar(_FallbackBar(label, max_val))
        _ACTIVE_PROGRESS_BAR = bar
        return bar
