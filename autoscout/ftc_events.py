import html
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


FTC_EVENTS_BASE = "https://ftc-events.firstinspires.org"
DEFAULT_TIMEOUT_S = 10.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 JuiceAutoScoutDashboard/1.0"
)
MATCH_LABEL_RE = re.compile(
    r"^(Qualification|Playoff|Quarterfinal|Semifinal|Final|Match)\s+(.+)$",
    re.IGNORECASE,
)
YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?[^\"'\s<]+|youtu\.be/[^\"'\s<]+)",
    re.IGNORECASE,
)
YOUTUBE_EMBED_RE = re.compile(
    r"https?://(?:www\.)?youtube\.com/embed/([^\"'\s<?&]+)",
    re.IGNORECASE,
)


@dataclass
class EventPage:
    phase: str
    url: str
    match_links: List[Tuple[str, str]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class MatchVideoRecord:
    phase: str
    label: str
    match_page_url: str
    red_teams: List[str] = field(default_factory=list)
    blue_teams: List[str] = field(default_factory=list)
    red_score: Optional[int] = None
    blue_score: Optional[int] = None
    winning_alliance: Optional[str] = None
    video_url: Optional[str] = None
    external_links: List[str] = field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None


def discover_event_videos(event_code: Optional[str] = None,
                          season: Optional[str] = None,
                          event_url: Optional[str] = None,
                          io_workers: int = 12) -> Dict[str, object]:
    parsed_season, parsed_code = _parse_event_reference(
        event_code=event_code,
        season=season,
        event_url=event_url,
    )
    event_root = "{}/{}/{}".format(FTC_EVENTS_BASE, parsed_season, parsed_code)
    pages = [
        EventPage("qualifications", event_root + "/qualifications"),
        EventPage("playoffs", event_root + "/playoffs"),
    ]

    event_title = None
    page_records_by_url: Dict[str, List[MatchVideoRecord]] = {}
    for page in pages:
        try:
            page_html = _fetch_text(page.url)
        except RuntimeError as exc:
            page.error = str(exc)
            continue
        if event_title is None:
            event_title = _extract_title(page_html)
        page_records = _extract_match_records(page_html, page.url, page.phase)
        page_records_by_url[page.url] = page_records
        page.match_links = [(record.label, record.match_page_url) for record in page_records]

    matches_by_url: Dict[str, MatchVideoRecord] = {}
    for page in pages:
        for record in page_records_by_url.get(page.url, []):
            if record.match_page_url not in matches_by_url:
                matches_by_url[record.match_page_url] = record

    if matches_by_url:
        max_workers = max(2, min(io_workers, 32))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ftc-scrape") as pool:
            future_map = {
                pool.submit(_discover_match_video, record.match_page_url): record
                for record in matches_by_url.values()
            }
            for future in as_completed(future_map):
                record = future_map[future]
                try:
                    video_url, external_links = future.result()
                    record.video_url = video_url
                    record.external_links = external_links
                    record.status = "video_found" if video_url else "no_video_found"
                except RuntimeError as exc:
                    record.status = "error"
                    record.error = str(exc)

    discovered_matches = sorted(matches_by_url.values(), key=_match_sort_key)
    return {
        "event_code": parsed_code,
        "season": parsed_season,
        "event_url": event_root,
        "title": event_title or parsed_code,
        "pages": [asdict(page) for page in pages],
        "matches": [asdict(match) for match in discovered_matches],
        "match_count": len(discovered_matches),
        "video_count": sum(1 for match in discovered_matches if match.video_url),
        "notes": [
            "FTC Events scraping is heuristic-based because match video links are not consistently exposed in the schedule HTML.",
            "When a direct YouTube URL is not discoverable, the dashboard still returns the match page URL so you can inspect it manually.",
        ],
    }


def _parse_event_reference(event_code: Optional[str],
                           season: Optional[str],
                           event_url: Optional[str]) -> Tuple[str, str]:
    if event_url:
        match = re.search(
            r"ftc-events\.firstinspires\.org/(\d{4})/([A-Z0-9]+)/",
            event_url,
            re.IGNORECASE,
        )
        if not match:
            raise RuntimeError("Could not parse season and event code from event URL.")
        return match.group(1), match.group(2).upper()

    if not event_code:
        raise RuntimeError("Provide either an event code or an FTC Events URL.")
    if not season:
        raise RuntimeError("A season year is required when using an event code directly.")
    return str(season), str(event_code).upper()


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError("HTTP {} while fetching {}".format(exc.code, url)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Network error while fetching {}: {}".format(url, exc.reason)) from exc


def _extract_title(page_html: str) -> Optional[str]:
    match = re.search(r"<title>(.*?)</title>", page_html, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _clean_text(match.group(1))


def _extract_match_records(page_html: str, page_url: str, phase: str) -> List[MatchVideoRecord]:
    anchors = _extract_anchor_matches(page_html, page_url)
    match_anchor_indexes = [
        index for index, anchor in enumerate(anchors)
        if MATCH_LABEL_RE.match(anchor["text"])
    ]
    records: List[MatchVideoRecord] = []
    seen = set()

    for list_index, anchor_index in enumerate(match_anchor_indexes):
        match_anchor = anchors[anchor_index]
        next_start = (
            anchors[match_anchor_indexes[list_index + 1]]["start"]
            if list_index + 1 < len(match_anchor_indexes)
            else len(page_html)
        )
        block_end_index = (
            match_anchor_indexes[list_index + 1]
            if list_index + 1 < len(match_anchor_indexes)
            else len(anchors)
        )
        block_anchors = anchors[anchor_index + 1:block_end_index]
        team_anchor_spans = []
        team_numbers: List[str] = []
        for anchor in block_anchors:
            text = anchor["text"]
            if text.isdigit():
                team_numbers.append(text)
                team_anchor_spans.append((anchor["start"], anchor["end"]))
            if len(team_numbers) >= 4:
                break

        trailing_start = team_anchor_spans[-1][1] if team_anchor_spans else match_anchor["end"]
        trailing_text = _clean_text(page_html[trailing_start:next_start])
        trailing_scores = [int(token) for token in re.findall(r"\b\d+\b", trailing_text)]
        red_score = trailing_scores[0] if len(trailing_scores) >= 1 else None
        blue_score = trailing_scores[1] if len(trailing_scores) >= 2 else None

        record = MatchVideoRecord(
            phase=phase,
            label=match_anchor["text"],
            match_page_url=_normalize_match_href(match_anchor["href"], phase),
            red_teams=team_numbers[:2],
            blue_teams=team_numbers[2:4],
            red_score=red_score,
            blue_score=blue_score,
            winning_alliance=_winning_alliance(red_score, blue_score),
        )
        key = (record.label, record.match_page_url)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def _extract_match_links(page_html: str, page_url: str, phase: str) -> List[Tuple[str, str]]:
    return [(record.label, record.match_page_url) for record in _extract_match_records(page_html, page_url, phase)]


def _extract_anchor_links(page_html: str, base_url: str) -> List[Tuple[str, str]]:
    anchors: List[Tuple[str, str]] = []
    for anchor in _extract_anchor_matches(page_html, base_url):
        anchors.append((anchor["href"], anchor["text"]))
    return anchors


def _extract_anchor_matches(page_html: str, base_url: str) -> List[Dict[str, object]]:
    anchors: List[Dict[str, object]] = []
    for match in re.finditer(
        r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        page_html,
        re.IGNORECASE | re.DOTALL,
    ):
        href = urllib.parse.urljoin(base_url, html.unescape(match.group(1)))
        text = _clean_text(match.group(2))
        if href and text:
            anchors.append({
                "href": href,
                "text": text,
                "start": match.start(),
                "end": match.end(),
            })
    return anchors


def _discover_match_video(match_page_url: str) -> Tuple[Optional[str], List[str]]:
    match_html = _fetch_text(match_page_url)
    external_links = []
    video_url = _extract_video_url(match_html)
    for href, text in _extract_anchor_links(match_html, match_page_url):
        if href.startswith(FTC_EVENTS_BASE):
            continue
        if href not in external_links:
            external_links.append(href)
        if video_url is None and _looks_like_video_link(href, text):
            video_url = href
    return video_url, external_links


def _extract_video_url(page_html: str) -> Optional[str]:
    match = YOUTUBE_URL_RE.search(page_html)
    if match:
        return _strip_trailing_punctuation(match.group(0))

    embed = YOUTUBE_EMBED_RE.search(page_html)
    if embed:
        return "https://www.youtube.com/watch?v={}".format(embed.group(1))

    generic_link = re.search(
        r"https?://[^\"'\s<]+(?:video|stream|watch)[^\"'\s<]*",
        page_html,
        re.IGNORECASE,
    )
    if generic_link:
        return _strip_trailing_punctuation(generic_link.group(0))
    return None


def _looks_like_video_link(href: str, text: str) -> bool:
    href_l = href.lower()
    text_l = text.lower()
    return (
        "youtube.com" in href_l
        or "youtu.be" in href_l
        or "watch" in href_l
        or "stream" in href_l
        or "video" in href_l
        or "watch" in text_l
        or "video" in text_l
        or "stream" in text_l
    )


def _clean_text(raw_text: str) -> str:
    stripped = re.sub(r"<[^>]+>", " ", raw_text)
    stripped = html.unescape(stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()


def _normalize_match_href(href: str, phase: str) -> str:
    if "/{}/".format(phase) in href:
        return href
    if href.endswith("/" + phase):
        return href
    return href


def _strip_trailing_punctuation(url: str) -> str:
    return url.rstrip(").,;\"'")


def _winning_alliance(red_score: Optional[int], blue_score: Optional[int]) -> Optional[str]:
    if red_score is None or blue_score is None:
        return None
    if red_score > blue_score:
        return "red"
    if blue_score > red_score:
        return "blue"
    return "tie"


def _match_sort_key(record: MatchVideoRecord) -> Tuple[int, int, str]:
    phase_order = {"qualifications": 0, "playoffs": 1}
    label_match = MATCH_LABEL_RE.match(record.label)
    match_number = 0
    if label_match:
        digits = re.findall(r"\d+", label_match.group(2))
        if digits:
            match_number = int(digits[0])
    return phase_order.get(record.phase, 99), match_number, record.label
