import importlib.util
import os
import re
import shutil
import ssl
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_COOKIE_FILE_ENV_VARS = ("AUTOSCOUT_YTDLP_COOKIES", "YTDLP_COOKIES")
_BROWSER_COOKIE_ENV_VARS = ("AUTOSCOUT_YTDLP_BROWSER_COOKIES", "YTDLP_BROWSER_COOKIES")
_EXTRACTOR_ARGS_ENV_VARS = ("AUTOSCOUT_YTDLP_EXTRACTOR_ARGS", "YTDLP_EXTRACTOR_ARGS")
_COOKIE_FILE_CANDIDATES = (
    "cookies.txt",
    ".cookies.txt",
    "yt-dlp-cookies.txt",
    ".yt-dlp-cookies.txt",
)
_SUPPORTED_BROWSERS = {"brave", "chrome", "chromium", "edge", "firefox", "safari"}
_POLITE_NETWORK_ARGS = [
    "--sleep-requests", "1",
    "--sleep-interval", "5",
    "--max-sleep-interval", "10",
    "--retry-sleep", "http:linear=5:15:5",
]
_LEGACY_MP4_DOWNLOAD_ARGS = ["-f", "best[ext=mp4]"]
_MP4_DOWNLOAD_ARGS = ["-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"]
_PLAYABLE_URL_ARGS = ["-f", "best[ext=mp4]/best", "-g"]
_WEB_SAFARI_DOWNLOAD_ARGS = ["--extractor-args", "youtube:player_client=web_safari", "-f", "b/best"]
_WEB_SAFARI_PLAYABLE_ARGS = ["--extractor-args", "youtube:player_client=web_safari", "-g"]


def find_yt_dlp_command() -> Optional[List[str]]:
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    yt_dlp_path = shutil.which("yt-dlp")
    if yt_dlp_path:
        return [yt_dlp_path]
    return None


def get_yt_dlp_command_warning(command: Sequence[str]) -> str:
    if len(command) != 1:
        return ""
    match = re.search(r"/Python/(\d+\.\d+)/bin/yt-dlp$", command[0])
    if not match:
        return ""
    current_version = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    installed_version = match.group(1)
    if installed_version == current_version:
        return ""
    return (
        "[WARN] yt-dlp is coming from a Python {} user install instead of the current Python {}. "
        "Reinstall it with `python3 -m pip install -U yt-dlp` to use the upgraded interpreter."
    ).format(installed_version, current_version)


def tls_help_message() -> str:
    major_minor = "{}.{}".format(sys.version_info.major, sys.version_info.minor)
    install_certificates_path = "/Applications/Python {}/Install Certificates.command".format(major_minor)
    if sys.platform == "darwin" and os.path.exists(install_certificates_path):
        return (
            "Your Python TLS trust store is not configured. Run `{}` and then retry. "
            "If needed, reinstall `yt-dlp` into this interpreter with `python3 -m pip install -U yt-dlp`."
        ).format(install_certificates_path)
    cafile = ssl.get_default_verify_paths().cafile
    if cafile:
        return (
            "Your Python TLS trust store is not configured correctly. Check the CA bundle at `{}` "
            "or reinstall `yt-dlp` into this interpreter with `python3 -m pip install -U yt-dlp`."
        ).format(cafile)
    return (
        "Your Python TLS trust store is not configured. Reinstall `yt-dlp` into this interpreter with "
        "`python3 -m pip install -U yt-dlp` and verify your system CA certificates."
    )


def yt_dlp_network_backoff_args() -> List[str]:
    return list(_POLITE_NETWORK_ARGS)


def output_indicates_rate_limit(output_lines: Sequence[str]) -> bool:
    joined_output = "\n".join(str(line).lower() for line in output_lines)
    return (
        "http error 429" in joined_output
        or "too many requests" in joined_output
        or "rate limit" in joined_output
    )


def rate_limit_help_message() -> str:
    return (
        "YouTube is rate-limiting this machine right now. Wait a bit before retrying, avoid starting several "
        "downloads back-to-back, and if the block keeps happening use authenticated yt-dlp access. "
        "{}"
    ).format(auth_help_message())


def build_download_attempts(
    search_roots: Sequence[Path],
    configured_cookie_path: Optional[str] = None,
    configured_extractor_args: Optional[str] = None,
    configured_browser_sources: Optional[str] = None,
) -> Tuple[List[Dict[str, List[str]]], List[str]]:
    attempts = [
        {
            "label": "legacy best mp4",
            "args": [
                *_LEGACY_MP4_DOWNLOAD_ARGS,
            ],
        },
        {
            "label": "yt-dlp default selection",
            "args": [],
        },
        {
            "label": "web_safari fallback",
            "args": [
                *_WEB_SAFARI_DOWNLOAD_ARGS,
            ],
        },
        {
            "label": "mp4 preferred",
            "args": [
                *_MP4_DOWNLOAD_ARGS,
            ],
        },
        {
            "label": "best mixed format",
            "args": [
                "-f",
                "bv*+ba/best",
            ],
        },
    ]
    return _append_auth_attempts(
        attempts,
        search_roots,
        _LEGACY_MP4_DOWNLOAD_ARGS,
        configured_cookie_path=configured_cookie_path,
        configured_extractor_args=configured_extractor_args,
        configured_browser_sources=configured_browser_sources,
    )


def build_playable_url_attempts(
    search_roots: Sequence[Path],
    configured_cookie_path: Optional[str] = None,
    configured_extractor_args: Optional[str] = None,
    configured_browser_sources: Optional[str] = None,
) -> Tuple[List[Dict[str, List[str]]], List[str]]:
    attempts = [
        {
            "label": "yt-dlp default selection",
            "args": ["-g"],
        },
        {
            "label": "web_safari fallback",
            "args": [
                *_WEB_SAFARI_PLAYABLE_ARGS,
            ],
        },
        {
            "label": "best mp4",
            "args": [
                *_PLAYABLE_URL_ARGS,
            ],
        },
        {
            "label": "best mixed format",
            "args": [
                "-f",
                "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
                "-g",
            ],
        },
    ]
    return _append_auth_attempts(
        attempts,
        search_roots,
        _PLAYABLE_URL_ARGS,
        configured_cookie_path=configured_cookie_path,
        configured_extractor_args=configured_extractor_args,
        configured_browser_sources=configured_browser_sources,
    )


def auth_help_message() -> str:
    return (
        "If YouTube now requires a signed-in session, export a Netscape cookies file from a private/incognito "
        "YouTube session and set AUTOSCOUT_YTDLP_COOKIES=/path/to/cookies.txt. If you have a PO token setup, "
        "you can also pass exact yt-dlp extractor args with AUTOSCOUT_YTDLP_EXTRACTOR_ARGS."
    )


def _append_auth_attempts(
    attempts: List[Dict[str, List[str]]],
    search_roots: Sequence[Path],
    format_args: List[str],
    configured_cookie_path: Optional[str] = None,
    configured_extractor_args: Optional[str] = None,
    configured_browser_sources: Optional[str] = None,
) -> Tuple[List[Dict[str, List[str]]], List[str]]:
    notes: List[str] = []
    configured_extractor_args = _configured_extractor_args(configured_extractor_args)
    cookie_file = _find_cookie_file(search_roots, configured_cookie_path)
    if configured_cookie_path and cookie_file is None:
        notes.append("[WARN] Configured yt-dlp cookies file was not found: {}".format(configured_cookie_path))
    if cookie_file is not None:
        attempts.append(
            {
                "label": "cookies file fallback",
                "args": ["--cookies", str(cookie_file), *format_args],
            }
        )
        notes.append("[INFO] Using yt-dlp cookies file fallback: {}".format(cookie_file))

    if configured_extractor_args:
        attempts.append(
            {
                "label": "custom extractor-args fallback",
                "args": ["--extractor-args", configured_extractor_args, *format_args],
            }
        )
        notes.append("[INFO] Using custom yt-dlp extractor args from environment.")

    browser_sources = _requested_browser_cookie_sources(configured_browser_sources)
    for browser in browser_sources:
        attempts.append(
            {
                "label": "{} cookies fallback".format(browser.title()),
                "args": ["--cookies-from-browser", browser, *format_args],
            }
        )

    if browser_sources:
        notes.append(
            "[INFO] Browser cookie fallback enabled via environment: {}".format(
                ", ".join(browser_sources)
            )
        )
    else:
        notes.append(
            "[INFO] Browser cookie fallback is disabled by default to avoid protected macOS cookie store access."
        )

    return attempts, notes


def _find_cookie_file(search_roots: Sequence[Path], configured_cookie_path: Optional[str] = None) -> Optional[Path]:
    if configured_cookie_path:
        candidate = Path(configured_cookie_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    for env_var in _COOKIE_FILE_ENV_VARS:
        configured = os.environ.get(env_var)
        if not configured:
            continue
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    seen = set()
    for root in search_roots:
        try:
            resolved_root = root.expanduser().resolve()
        except OSError:
            continue
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        for name in _COOKIE_FILE_CANDIDATES:
            candidate = resolved_root / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def _requested_browser_cookie_sources(configured_browser_sources: Optional[str] = None) -> List[str]:
    raw = ""
    if configured_browser_sources is not None:
        raw = configured_browser_sources.strip()
    for env_var in _BROWSER_COOKIE_ENV_VARS:
        if raw:
            break
        raw = os.environ.get(env_var, "").strip()
        if raw:
            break
    if not raw:
        return []

    browsers: List[str] = []
    for item in raw.split(","):
        browser = item.strip().lower()
        if not browser or browser in browsers:
            continue
        if browser in _SUPPORTED_BROWSERS:
            browsers.append(browser)
    return browsers


def _configured_extractor_args(configured_extractor_args: Optional[str] = None) -> str:
    if configured_extractor_args:
        return configured_extractor_args.strip()
    for env_var in _EXTRACTOR_ARGS_ENV_VARS:
        raw = os.environ.get(env_var, "").strip()
        if raw:
            return raw
    return ""
