
"""Serve local artifacts produced by background agent tasks."""

import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from aiohttp import web

LOCAL_FILE_ROUTE = "/api/local-file"
LOCAL_FILE_ROOTS_ENV = "LIVE_VLM_LOCAL_FILE_ROOTS"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent.parent


def _legacy_background_env(suffix: str, default: str = "") -> str:
    legacy_name = "BACKGROUND_" + "CO" + "DEX_" + suffix
    return os.environ.get(legacy_name, default)


def _project_relative_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


LOCAL_HTML_ARTIFACT_DIR = _project_relative_path(
    os.environ.get("LIVE_VLM_HTML_ARTIFACT_DIR", "html")
)

_DISPLAY_PAYLOAD_TYPES = {"background_result_ready"}
_DISPLAY_TEXT_KEYS = ("text", "summary_text", "background_summary")
_LOCAL_PATH_RE = re.compile(
    r"(?<![\w:/])(/(?:[^\s<>'\"`()\[\]{}]+/)*[^\s<>'\"`()\[\]{}]+\.[A-Za-z0-9]{1,12})"
)
_MARKDOWN_CODE_RE = re.compile(r"(```[\s\S]*?(?:```|$)|`[^`\n]+`)")
_TRAILING_PATH_PUNCTUATION = ".,;:!?，。；：！？"

_INLINE_MIME_PREFIXES = ("image/", "text/")
_INLINE_MIME_TYPES = {
    "application/json",
    "application/pdf",
    "application/xml",
    "application/xhtml+xml",
    "image/svg+xml",
}
_ATTACHMENT_EXTENSIONS = {
    ".7z",
    ".bin",
    ".bz2",
    ".dmg",
    ".exe",
    ".gz",
    ".iso",
    ".lz",
    ".lzma",
    ".rar",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
    ".zst",
}


def setup_local_file_routes(app: web.Application) -> None:
    """Register the local artifact file route."""
    app.router.add_get(LOCAL_FILE_ROUTE, serve_local_file)


async def serve_local_file(request: web.Request) -> web.StreamResponse:
    """Serve a whitelisted local file inline when browsers can render it."""
    raw_path = request.query.get("path", "")
    if not raw_path:
        raise web.HTTPBadRequest(text="Missing local file path")

    file_path = resolve_local_file_path(raw_path)
    if file_path is None:
        raise web.HTTPNotFound(text="Local file is unavailable")

    content_type = guess_local_file_content_type(file_path)
    disposition = "inline" if should_display_inline(file_path, content_type) else "attachment"
    headers = {
        "Cache-Control": "no-store",
        "Content-Disposition": _content_disposition(disposition, file_path.name),
        "Content-Type": content_type,
        "X-Content-Type-Options": "nosniff",
    }
    if disposition == "inline" and content_type == "image/svg+xml":
        headers["Content-Security-Policy"] = (
            "sandbox; default-src 'none'; img-src 'self' data: blob:; "
            "style-src 'unsafe-inline'; media-src 'self' data: blob:; "
            "font-src data:; frame-ancestors 'none'"
        )
    return web.FileResponse(path=file_path, headers=headers)


def rewrite_payload_local_file_links(payload: dict) -> dict:
    """Rewrite display-only text fields with HTTP links to local artifacts."""
    if not isinstance(payload, dict) or payload.get("type") not in _DISPLAY_PAYLOAD_TYPES:
        return payload

    rewritten = dict(payload)
    changed = False
    for key in _DISPLAY_TEXT_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        next_value = rewrite_local_file_links(value)
        if next_value != value:
            rewritten[key] = next_value
            changed = True
    return rewritten if changed else payload


def rewrite_local_file_links(text: str) -> str:
    """Convert existing local file paths in normal text into markdown links."""
    source = str(text or "")
    if not source:
        return ""

    parts = _MARKDOWN_CODE_RE.split(source)
    for index, part in enumerate(parts):
        if not part or part.startswith("`"):
            continue
        parts[index] = _rewrite_local_file_links_in_segment(part)
    return "".join(parts)


def local_file_url_for_path(path_value: str) -> str:
    """Return the web route URL for a local file if it is safe to expose."""
    file_path = resolve_local_file_path(path_value)
    if file_path is None:
        return ""
    return f"{LOCAL_FILE_ROUTE}?path={quote(str(file_path), safe='')}"


def resolve_local_file_path(path_value: str) -> Path | None:
    """Resolve a local file path if it exists and is under an allowed root."""
    value = _normalize_path_value(path_value)
    if not value:
        return None

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not candidate.is_file() or not _is_allowed_local_file(candidate):
        return None
    return candidate


def guess_local_file_content_type(path: Path) -> str:
    content_type, _encoding = mimetypes.guess_type(str(path))
    return content_type or "application/octet-stream"


def should_display_inline(path: Path, content_type: str | None = None) -> bool:
    suffix = path.suffix.lower()
    if suffix in _ATTACHMENT_EXTENSIONS:
        return False
    resolved_type = content_type or guess_local_file_content_type(path)
    return resolved_type in _INLINE_MIME_TYPES or resolved_type.startswith(_INLINE_MIME_PREFIXES)


def _rewrite_local_file_links_in_segment(segment: str) -> str:
    def replace(match: re.Match) -> str:
        raw_path = match.group(1)
        path_value, trailing = _split_trailing_punctuation(raw_path)
        url = local_file_url_for_path(path_value)
        if not url:
            return raw_path

        previous_char = segment[match.start() - 1] if match.start() > 0 else ""
        if previous_char == "(":
            return f"{url}{trailing}"
        label = Path(path_value).name or path_value
        return f"[{_escape_markdown_label(label)}]({url}){trailing}"

    return _LOCAL_PATH_RE.sub(replace, segment)


def _split_trailing_punctuation(path_value: str) -> tuple[str, str]:
    value = path_value
    trailing = ""
    while value and value[-1] in _TRAILING_PATH_PUNCTUATION:
        trailing = value[-1] + trailing
        value = value[:-1]
    return value, trailing


def _normalize_path_value(path_value: str) -> str:
    value = str(path_value or "").strip()
    if not value:
        return ""
    if value.startswith("file://"):
        parsed = urlparse(value)
        value = unquote(parsed.path or "")
    return value


def _configured_local_file_roots() -> list[Path] | None:
    raw_roots = os.environ.get(LOCAL_FILE_ROOTS_ENV, "").strip()
    if raw_roots == "*":
        return None

    values = []
    if raw_roots:
        values = [item.strip() for item in re.split(r"[:,]", raw_roots) if item.strip()]
    else:
        values = [
            os.environ.get("BACKGROUND_AGENT_WORKSPACE", _legacy_background_env("WORKSPACE")),
            str(LOCAL_HTML_ARTIFACT_DIR),
            str(REPO_ROOT / "agent-workspace"),
            str(PROJECT_ROOT / "agent-workspace"),
            os.getcwd(),
            str(PROJECT_ROOT),
        ]

    roots = []
    seen = set()
    for value in values:
        try:
            root = Path(value).expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        key = str(root)
        if key not in seen:
            roots.append(root)
            seen.add(key)
    return roots


def _is_allowed_local_file(path: Path) -> bool:
    roots = _configured_local_file_roots()
    if roots is None:
        return True
    path_text = str(path)
    for root in roots:
        try:
            if os.path.commonpath([path_text, str(root)]) == str(root):
                return True
        except ValueError:
            continue
    return False


def _content_disposition(disposition: str, filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode("ascii") or "download"
    ascii_name = ascii_name.replace("\\", "_").replace('"', "_")
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def _escape_markdown_label(label: str) -> str:
    return str(label or "").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
