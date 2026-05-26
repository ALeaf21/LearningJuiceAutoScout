import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from autoscout.ftc_events import discover_event_videos
from autoscout.hardware import collect_hardware_profile, collect_runtime_usage
from autoscout.ytdlp import (
    auth_help_message,
    build_playable_url_attempts,
    find_yt_dlp_command,
    get_yt_dlp_command_warning,
    output_indicates_rate_limit,
    rate_limit_help_message,
    yt_dlp_network_backoff_args,
)


PROCESS_STAGE_ORDER = [
    "Loading Match Footage",
    "Calibrating Trackers",
    "Tracking Match",
    "Cleaning Up",
]
DASHBOARD_EVENT_PREFIX = "@@AUTOSCOUT_DASHBOARD@@"


def _initial_process_stages() -> Dict[str, Dict[str, object]]:
    return {
        name: {
            "name": name,
            "status": "pending",
            "detail": "",
            "progress_current": None,
            "progress_total": None,
            "progress_fraction": None,
            "stage_started_at": None,
            "stage_completed_at": None,
        }
        for name in PROCESS_STAGE_ORDER
    }


@dataclass
class TrackJob:
    job_id: str
    command: List[str]
    cwd: str
    output_dir: str
    source_type: str
    source_url: Optional[str] = None
    source_video_path: Optional[str] = None
    event_code: Optional[str] = None
    match_label: Optional[str] = None
    match_phase: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"
    return_code: Optional[int] = None
    pid: Optional[int] = None
    logs: deque = field(default_factory=lambda: deque(maxlen=1200))
    process_stages: Dict[str, Dict[str, object]] = field(default_factory=_initial_process_stages)
    preview_frame_count: int = 0
    latest_preview_bytes: Optional[bytes] = None
    latest_preview_content_type: str = "image/jpeg"
    latest_preview_source: Optional[str] = None
    latest_preview_updated_at: Optional[float] = None
    stop_requested: bool = False
    process: Optional[subprocess.Popen] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "command": self.command,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "return_code": self.return_code,
            "pid": self.pid,
            "output_dir": self.output_dir,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "source_video_path": self.source_video_path,
            "event_code": self.event_code,
            "match_label": self.match_label,
            "match_phase": self.match_phase,
            "log_tail": list(self.logs)[-120:],
            "process_stages": [dict(self.process_stages[name]) for name in PROCESS_STAGE_ORDER],
            "preview_frame_count": self.preview_frame_count,
            "stop_requested": self.stop_requested,
        }


class DashboardState:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()
        self.jobs: Dict[str, TrackJob] = {}
        self.jobs_lock = threading.Lock()
        self.hardware_lock = threading.Lock()
        self.hardware_profile = collect_hardware_profile()
        self.settings_lock = threading.Lock()
        self.settings = {
            "scrape_workers_override": None,
            "parallel_track_jobs_override": None,
            "default_debug_frames": True,
            "default_debug_video": False,
            "default_debug_every": 1,
            "auto_open_process_view": True,
            "output_root": "./output_dashboard",
            "ytdlp_cookies_path": "",
            "ytdlp_extractor_args": "",
        }

    def refresh_hardware(self) -> Dict[str, object]:
        with self.hardware_lock:
            self.hardware_profile = collect_hardware_profile()
            return self.hardware_profile

    def get_hardware(self) -> Dict[str, object]:
        with self.hardware_lock:
            return self.hardware_profile

    def get_runtime_usage(self) -> Dict[str, object]:
        return collect_runtime_usage()

    def resolve_video_source(self, payload: Dict[str, object]) -> Dict[str, object]:
        workspace_url = payload.get("workspace_url")
        youtube_url = payload.get("youtube_url")
        if workspace_url:
            if not isinstance(workspace_url, str) or not workspace_url.startswith("/workspace/"):
                raise RuntimeError("Invalid workspace video URL.")
            return {
                "kind": "workspace",
                "playable_url": workspace_url,
            }
        if youtube_url:
            settings = self.get_settings()
            playable_url = _resolve_youtube_playable_url(
                str(youtube_url),
                configured_cookie_path=str(settings.get("ytdlp_cookies_path") or ""),
                configured_extractor_args=str(settings.get("ytdlp_extractor_args") or ""),
            )
            return {
                "kind": "youtube",
                "playable_url": playable_url,
            }
        raise RuntimeError("Provide either a workspace_url or a youtube_url.")

    def get_settings(self) -> Dict[str, object]:
        with self.settings_lock:
            return dict(self.settings)

    def update_settings(self, updates: Dict[str, object]) -> Dict[str, object]:
        allowed_keys = set(self.settings.keys())
        with self.settings_lock:
            for key, value in updates.items():
                if key in allowed_keys:
                    self.settings[key] = value
            return dict(self.settings)

    def list_examples(self) -> Dict[str, object]:
        examples_dir = self.root_dir / "examples"
        videos = []
        if examples_dir.exists():
            for path in sorted(examples_dir.glob("*.mp4")):
                rel = path.resolve().relative_to(self.root_dir)
                videos.append({
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "workspace_url": "/workspace/" + str(rel).replace(os.sep, "/"),
                })
        corners_path = self.root_dir / "field_corners.json"
        return {
            "examples": videos,
            "default_corners_path": str(corners_path) if corners_path.exists() else None,
            "project_root": str(self.root_dir),
        }

    def list_downloaded_videos(self, event_code: str) -> Dict[str, object]:
        normalized_event_code = str(event_code or "").strip().upper()
        if not normalized_event_code:
            return {"event_code": "", "videos": []}
        event_root = self._event_output_root(normalized_event_code)
        videos: List[Dict[str, object]] = []
        if event_root.exists():
            for video_path in sorted(event_root.glob("*/match_video.mp4")):
                workspace_url = self._workspace_url(video_path)
                if not workspace_url:
                    continue
                videos.append({
                    "workspace_url": workspace_url,
                    "match_slug": video_path.parent.name,
                    "output_dir": str(video_path.parent),
                    "video_path": str(video_path),
                    "modified_at": video_path.stat().st_mtime,
                })
        return {
            "event_code": normalized_event_code,
            "videos": videos,
        }

    def list_event_artifacts(self, event_code: str) -> Dict[str, object]:
        normalized_event_code = str(event_code or "").strip().upper()
        if not normalized_event_code:
            return {"event_code": "", "artifacts": []}
        event_root = self._event_output_root(normalized_event_code)
        artifacts: List[Dict[str, object]] = []
        if event_root.exists():
            for match_dir in sorted(path for path in event_root.iterdir() if path.is_dir()):
                artifact = self._stored_match_artifact(match_dir)
                if artifact is None:
                    continue
                artifacts.append(artifact)
        return {
            "event_code": normalized_event_code,
            "artifacts": artifacts,
        }

    def save_corners(self, filename: str, corners_px: List[List[float]]) -> Dict[str, object]:
        if not filename:
            raise RuntimeError("A filename is required.")
        destination = (self.root_dir / filename).resolve()
        if self.root_dir not in destination.parents and destination != self.root_dir:
            raise RuntimeError("Corner files must be saved inside the project workspace.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"corners_px": corners_px}
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        return {"saved": True, "path": str(destination)}

    def start_track_job(self, payload: Dict[str, object]) -> Dict[str, object]:
        source_type = str(payload.get("source_type") or "youtube").lower()
        settings = self.get_settings()
        output_dir = str(payload.get("output_dir") or settings.get("output_root") or "./output")
        resolved_output_dir = Path(output_dir).expanduser()
        if not resolved_output_dir.is_absolute():
            resolved_output_dir = (self.root_dir / resolved_output_dir).resolve()
        command = [sys.executable, "auto_scout.py", "--output-dir", output_dir]
        command.append("--auto-match-bounds")

        corners = payload.get("corners")
        if corners:
            command.extend(["--corners", str(corners)])

        start_offset = payload.get("start_offset")
        if start_offset not in (None, ""):
            command.extend(["--start-offset", str(start_offset)])

        sample_rate = payload.get("sample_rate")
        if sample_rate not in (None, ""):
            command.extend(["--sample-rate", str(sample_rate)])

        debug_every = payload.get("debug_every")
        if debug_every in (None, ""):
            debug_every = settings.get("default_debug_every")
        command.extend(["--debug-every", str(debug_every)])

        debug_enabled = bool(payload.get("debug")) if payload.get("debug") is not None else bool(settings.get("default_debug_frames"))
        debug_video = bool(payload.get("debug_video")) if payload.get("debug_video") is not None else bool(settings.get("default_debug_video"))
        if debug_enabled:
            command.append("--debug")
        if debug_video:
            command.append("--debug-video")
        if payload.get("debug_enable_hitboxes"):
            command.append("--debug-enable-hitboxes")

        manual_reference_csv = payload.get("manual_reference_csv")
        if manual_reference_csv:
            command.extend(["--manual-reference-csv", str(manual_reference_csv)])

        robot_init_positions = payload.get("robot_init_positions")
        if robot_init_positions:
            if isinstance(robot_init_positions, str):
                robot_init_text = robot_init_positions
            else:
                robot_init_text = json.dumps(robot_init_positions)
            command.extend(["--robot-init-positions", robot_init_text])

        resolved_source_video_path = None
        if source_type == "local":
            video_path = payload.get("video_path")
            if not video_path:
                raise RuntimeError("A local video path is required.")
            command.extend(["--no-download", "--video-path", str(video_path)])
            resolved_source_video_path = str(video_path)
        else:
            url = payload.get("url")
            if not url:
                raise RuntimeError("A YouTube URL is required.")
            cached_video_path = resolved_output_dir / "match_video.mp4"
            if cached_video_path.exists():
                command.extend(["--no-download", "--video-path", str(cached_video_path)])
                resolved_source_video_path = str(cached_video_path)
            else:
                command.append(str(url))

        job = TrackJob(
            job_id=uuid.uuid4().hex[:10],
            command=command,
            cwd=str(self.root_dir),
            output_dir=str(resolved_output_dir),
            source_type=source_type,
            source_url=str(payload.get("url")) if payload.get("url") else None,
            source_video_path=resolved_source_video_path,
            event_code=str(payload.get("event_code")) if payload.get("event_code") else None,
            match_label=str(payload.get("match_label")) if payload.get("match_label") else None,
            match_phase=str(payload.get("match_phase")) if payload.get("match_phase") else None,
        )
        with self.jobs_lock:
            self.jobs[job.job_id] = job
        thread = threading.Thread(target=self._run_track_job, args=(job,), daemon=True)
        thread.start()
        return job.to_dict()

    def stop_job(self, job_id: str) -> Dict[str, object]:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise RuntimeError("Job not found.")
            process = job.process
            if process is None or process.poll() is not None:
                raise RuntimeError("Job is not currently running.")
            if job.status != "stopping":
                job.status = "stopping"
            job.stop_requested = True
            job.logs.append("[INFO] Stop requested for job {}.".format(job.job_id))
        try:
            process.terminate()
        except OSError as exc:
            raise RuntimeError("Failed to stop job: {}".format(exc)) from exc
        payload = self.get_job(job_id)
        if payload is None:
            raise RuntimeError("Job not found.")
        return payload

    def _run_track_job(self, job: TrackJob) -> None:
        env = os.environ.copy()
        recommendations = self.get_hardware().get("recommendations") or {}
        suggested_env = recommendations.get("suggested_env") or {}
        for key, value in suggested_env.items():
            env.setdefault(key, str(value))
        settings = self.get_settings()
        configured_cookie_path = str(settings.get("ytdlp_cookies_path") or "").strip()
        configured_extractor_args = str(settings.get("ytdlp_extractor_args") or "").strip()
        if configured_cookie_path:
            env["AUTOSCOUT_YTDLP_COOKIES"] = configured_cookie_path
        if configured_extractor_args:
            env["AUTOSCOUT_YTDLP_EXTRACTOR_ARGS"] = configured_extractor_args
        env["AUTOSCOUT_DASHBOARD_MODE"] = "1"
        env.setdefault("AUTOSCOUT_DASHBOARD_PREVIEW_EVERY", "5")

        job.status = "running"
        job.started_at = time.time()
        try:
            process = subprocess.Popen(
                job.command,
                cwd=job.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            job.logs.append("[ERROR] Failed to launch job: {}".format(exc))
            job.status = "failed"
            job.return_code = -1
            job.finished_at = time.time()
            return

        job.process = process
        job.pid = process.pid
        job.logs.append("[INFO] Started job {} (pid={})".format(job.job_id, job.pid))

        if process.stdout:
            for line in process.stdout:
                self._consume_process_output(job, line.rstrip())

        return_code = process.wait()
        job.return_code = return_code
        job.finished_at = time.time()
        if job.stop_requested and return_code != 0:
            job.status = "cancelled"
            self._fail_in_progress_stages(job, "Stopped")
        else:
            job.status = "completed" if return_code == 0 else "failed"
            if job.status == "failed":
                self._fail_in_progress_stages(job, "Failed")
        job.logs.append("[DONE] Job finished with code {}".format(return_code))
        if job.status == "completed":
            self._delete_downloaded_video(Path(job.output_dir), job.logs)

    def list_jobs(self) -> List[Dict[str, object]]:
        with self.jobs_lock:
            jobs = sorted(self.jobs.values(), key=lambda item: item.created_at, reverse=True)
            payload = []
            for job in jobs:
                item = job.to_dict()
                item.update(self._job_artifacts(job))
                payload.append(item)
            return payload

    def get_job(self, job_id: str) -> Optional[Dict[str, object]]:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            payload = job.to_dict()
            payload.update(self._job_artifacts(job))
            return payload

    def get_job_preview_asset(self, job_id: str) -> Optional[Dict[str, object]]:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if job.latest_preview_bytes is not None:
                return {
                    "content_type": job.latest_preview_content_type,
                    "raw": job.latest_preview_bytes,
                }
            preview_path = self._latest_preview_path(job)
            if preview_path is None or not preview_path.exists():
                return None
            return {
                "content_type": mimetypes.guess_type(str(preview_path))[0] or "application/octet-stream",
                "path": preview_path,
            }

    def _job_artifacts(self, job: TrackJob) -> Dict[str, object]:
        preview_path = self._latest_preview_path(job)
        debug_dir = Path(job.output_dir) / "tracker_debug"
        debug_frames_count = job.preview_frame_count
        if debug_frames_count == 0 and debug_dir.exists():
            debug_frames_count = len(list(debug_dir.glob("*.jpg")))
        csv_path = Path(job.output_dir) / "robot_positions.csv"
        jlog_path = Path(job.output_dir) / "robot_positions.jlog"
        wpilog_path = Path(job.output_dir) / "match_log.wpilog"
        downloaded_video_path = Path(job.output_dir) / "match_video.mp4"
        payload: Dict[str, object] = {
            "debug_frames_count": debug_frames_count,
            "latest_preview_url": None,
            "latest_preview_path": job.latest_preview_source or (str(preview_path) if preview_path else None),
            "workspace_output_url": self._workspace_url(Path(job.output_dir)),
            "source_workspace_url": self._workspace_url(Path(job.source_video_path)) if job.source_video_path else None,
            "downloaded_video_workspace_url": self._workspace_url(downloaded_video_path) if downloaded_video_path.exists() else None,
            "csv_workspace_url": self._workspace_url(csv_path) if csv_path.exists() else None,
            "jlog_workspace_url": self._workspace_url(jlog_path) if jlog_path.exists() else None,
            "wpilog_workspace_url": self._workspace_url(wpilog_path) if wpilog_path.exists() else None,
        }
        if job.latest_preview_bytes is not None or preview_path is not None:
            ts_value = int((job.latest_preview_updated_at or time.time()) * 1000)
            payload["latest_preview_url"] = "/api/jobs/{}/preview?ts={}".format(job.job_id, ts_value)
        return payload

    def _event_output_root(self, normalized_event_code: str) -> Path:
        settings = self.get_settings()
        output_root = Path(str(settings.get("output_root") or "./output_dashboard")).expanduser()
        if not output_root.is_absolute():
            output_root = (self.root_dir / output_root).resolve()
        return output_root / normalized_event_code

    def _stored_match_artifact(self, match_dir: Path) -> Optional[Dict[str, object]]:
        csv_path = match_dir / "robot_positions.csv"
        jlog_path = match_dir / "robot_positions.jlog"
        wpilog_path = match_dir / "match_log.wpilog"
        if not csv_path.exists() and not jlog_path.exists() and not wpilog_path.exists():
            return None
        preview_path = self._latest_preview_path_for_output_dir(match_dir)
        preview_workspace_url = self._workspace_url(preview_path) if preview_path else None
        preview_frames_count = 0
        debug_dir = match_dir / "tracker_debug"
        if debug_dir.exists():
            preview_frames_count = len(list(debug_dir.glob("*.jpg")))
        modified_candidates = [path for path in [csv_path, jlog_path, wpilog_path, preview_path] if path and path.exists()]
        modified_at = max((path.stat().st_mtime for path in modified_candidates), default=match_dir.stat().st_mtime)
        return {
            "match_slug": match_dir.name,
            "output_dir": str(match_dir),
            "workspace_output_url": self._workspace_url(match_dir),
            "csv_workspace_url": self._workspace_url(csv_path) if csv_path.exists() else None,
            "jlog_workspace_url": self._workspace_url(jlog_path) if jlog_path.exists() else None,
            "wpilog_workspace_url": self._workspace_url(wpilog_path) if wpilog_path.exists() else None,
            "latest_preview_url": preview_workspace_url,
            "latest_preview_path": str(preview_path) if preview_path else None,
            "debug_frames_count": preview_frames_count,
            "modified_at": modified_at,
        }

    def _latest_preview_path(self, job: TrackJob) -> Optional[Path]:
        return self._latest_preview_path_for_output_dir(Path(job.output_dir))

    def _latest_preview_path_for_output_dir(self, output_dir: Path) -> Optional[Path]:
        debug_dir = output_dir / "tracker_debug"
        if debug_dir.exists():
            frames = sorted(debug_dir.glob("*.jpg"))
            if frames:
                return frames[-1]
        debug_video = output_dir / "tracker_debug.mp4"
        if debug_video.exists():
            return debug_video
        return None

    def _consume_process_output(self, job: TrackJob, line: str) -> None:
        if line.startswith(DASHBOARD_EVENT_PREFIX):
            raw_payload = line[len(DASHBOARD_EVENT_PREFIX):]
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                return
            self._apply_dashboard_event(job, payload)
            return
        job.logs.append(line)

    def _apply_dashboard_event(self, job: TrackJob, payload: Dict[str, object]) -> None:
        kind = payload.get("kind")
        if kind == "stage":
            stage_name = str(payload.get("stage") or "")
            if stage_name in job.process_stages:
                stage = job.process_stages[stage_name]
                previous_status = str(stage.get("status") or "pending")
                next_status = str(payload.get("status") or "pending")
                now = time.time()
                if next_status == "in_progress" and previous_status != "in_progress":
                    stage["stage_started_at"] = now
                    stage["stage_completed_at"] = None
                elif next_status == "completed":
                    if stage.get("stage_started_at") is None:
                        stage["stage_started_at"] = now
                    stage["stage_completed_at"] = now
                stage["status"] = next_status
                stage["detail"] = str(payload.get("detail") or "")
                stage["progress_current"] = payload.get("progress_current")
                stage["progress_total"] = payload.get("progress_total")
                stage["progress_fraction"] = payload.get("progress_fraction")
                if stage["status"] == "completed":
                    stage["progress_current"] = 1
                    stage["progress_total"] = 1
                    stage["progress_fraction"] = 1.0
                elif stage["status"] != "in_progress":
                    stage["progress_current"] = None
                    stage["progress_total"] = None
                    stage["progress_fraction"] = None
            return
        if kind == "preview":
            raw_b64 = payload.get("jpeg_b64")
            if not isinstance(raw_b64, str) or not raw_b64:
                return
            try:
                job.latest_preview_bytes = base64.b64decode(raw_b64)
            except (ValueError, TypeError):
                return
            job.latest_preview_content_type = "image/jpeg"
            job.latest_preview_source = "dashboard-stream"
            job.latest_preview_updated_at = time.time()
            job.preview_frame_count += 1

    def _fail_in_progress_stages(self, job: TrackJob, detail: str) -> None:
        for stage_name in PROCESS_STAGE_ORDER:
            stage = job.process_stages[stage_name]
            if stage["status"] == "in_progress":
                stage["status"] = "failed"
                if detail:
                    stage["detail"] = detail

    def _workspace_url(self, path: Path) -> Optional[str]:
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(self.root_dir)
        except (OSError, ValueError):
            return None
        return "/workspace/" + str(rel).replace(os.sep, "/")

    def _delete_downloaded_video(self, output_dir: Path, logs: deque) -> None:
        video_path = output_dir / "match_video.mp4"
        archive_path = output_dir / "match_video.mp4.xz"
        removed_any = False
        try:
            if video_path.exists():
                video_path.unlink()
                removed_any = True
            if archive_path.exists():
                archive_path.unlink()
                removed_any = True
            if removed_any:
                logs.append("[INFO] Deleted cached downloaded match video after successful processing.")
        except OSError as exc:
            logs.append("[WARN] Failed to delete downloaded match video: {}".format(exc))


class DashboardHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, root_dir: Path):
        super().__init__(server_address, handler_class)
        self.root_dir = root_dir.resolve()
        self.static_dir = self.root_dir / "dashboard_static"
        self.state = DashboardState(self.root_dir)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            self._write_json({"ok": True, "time": time.time()})
            return
        if path == "/api/hardware":
            self._write_json(self.server.state.get_hardware())
            return
        if path == "/api/hardware/refresh":
            self._write_json(self.server.state.refresh_hardware())
            return
        if path == "/api/usage":
            self._write_json(self.server.state.get_runtime_usage())
            return
        if path == "/api/settings":
            self._write_json(self.server.state.get_settings())
            return
        if path == "/api/examples":
            self._write_json(self.server.state.list_examples())
            return
        if path == "/api/event/downloads":
            event_code = parse_qs(parsed.query).get("event_code", [""])[0]
            self._write_json(self.server.state.list_downloaded_videos(event_code))
            return
        if path == "/api/event/artifacts":
            event_code = parse_qs(parsed.query).get("event_code", [""])[0]
            self._write_json(self.server.state.list_event_artifacts(event_code))
            return
        if path == "/api/jobs":
            self._write_json({"jobs": self.server.state.list_jobs()})
            return
        if re.fullmatch(r"/api/jobs/[^/]+/preview", path):
            job_id = path.split("/")[3]
            preview_asset = self.server.state.get_job_preview_asset(job_id)
            if preview_asset is None:
                self._write_json({"error": "Preview not available."}, status=HTTPStatus.NOT_FOUND)
                return
            if "raw" in preview_asset:
                self._write_raw(preview_asset["raw"], str(preview_asset.get("content_type") or "application/octet-stream"))
            else:
                self._serve_path(preview_asset["path"])
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            payload = self.server.state.get_job(job_id)
            if payload is None:
                self._write_json({"error": "Job not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._write_json(payload)
            return

        if path.startswith("/tools/") or path.startswith("/assets/") or path.startswith("/util/"):
            self._serve_project_file(path.lstrip("/"))
            return
        if path.startswith("/workspace/"):
            self._serve_project_file(path[len("/workspace/"):])
            return
        if path == "/" or path == "/index.html":
            self._serve_static_file("index.html")
            return
        self._serve_static_file(path.lstrip("/"))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._read_json_body()
        path = parsed.path

        try:
            if path == "/api/event/discover":
                season = body.get("season")
                event_code = body.get("event_code")
                event_url = body.get("event_url")
                io_workers = body.get("io_workers")
                if io_workers in (None, ""):
                    io_workers = self.server.state.get_hardware()["recommendations"]["recommended_scrape_workers"]
                result = discover_event_videos(
                    event_code=event_code,
                    season=season,
                    event_url=event_url,
                    io_workers=int(io_workers),
                )
                self._write_json(result)
                return

            if path == "/api/jobs/start-track":
                payload = self.server.state.start_track_job(body)
                self._write_json(payload, status=HTTPStatus.CREATED)
                return

            if re.fullmatch(r"/api/jobs/[^/]+/stop", path):
                job_id = path.split("/")[3]
                payload = self.server.state.stop_job(job_id)
                self._write_json(payload)
                return

            if path == "/api/resolve-video-source":
                payload = self.server.state.resolve_video_source(body)
                self._write_json(payload)
                return

            if path == "/api/settings":
                payload = self.server.state.update_settings(body)
                self._write_json(payload)
                return

            if path == "/api/save-corners":
                filename = str(body.get("filename") or "field_corners.json")
                corners_px = body.get("corners_px")
                if not isinstance(corners_px, list) or len(corners_px) != 4:
                    raise RuntimeError("corners_px must contain exactly four [x, y] points.")
                result = self.server.state.save_corners(filename, corners_px)
                self._write_json(result, status=HTTPStatus.CREATED)
                return
        except RuntimeError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._write_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return

    def _serve_static_file(self, relative_path: str) -> None:
        static_path = (self.server.static_dir / relative_path).resolve()
        if not static_path.exists() or not static_path.is_file():
            self._write_json({"error": "Static file not found."}, status=HTTPStatus.NOT_FOUND)
            return
        if self.server.static_dir not in static_path.parents and static_path != self.server.static_dir:
            self._write_json({"error": "Invalid static file path."}, status=HTTPStatus.FORBIDDEN)
            return
        self._serve_path(static_path)

    def _serve_project_file(self, relative_path: str) -> None:
        file_path = (self.server.root_dir / relative_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            self._write_json({"error": "Project file not found."}, status=HTTPStatus.NOT_FOUND)
            return
        if self.server.root_dir not in file_path.parents and file_path != self.server.root_dir:
            self._write_json({"error": "Invalid project file path."}, status=HTTPStatus.FORBIDDEN)
            return
        self._serve_path(file_path)

    def _serve_path(self, file_path: Path) -> None:
        mime_type, _encoding = mimetypes.guess_type(str(file_path))
        content_type = mime_type or "application/octet-stream"
        try:
            total_size = file_path.stat().st_size
        except OSError as exc:
            self._write_json({"error": "Failed to read file: {}".format(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        range_header = self.headers.get("Range")
        if range_header:
            try:
                start, end = self._parse_byte_range(range_header, total_size)
            except ValueError:
                self._write_range_not_satisfiable(total_size)
                return
            self._send_file_bytes(file_path, content_type, start, end, total_size, HTTPStatus.PARTIAL_CONTENT)
            return
        end = max(total_size - 1, 0)
        self._send_file_bytes(file_path, content_type, 0, end, total_size, HTTPStatus.OK)

    def _write_raw(self, raw: bytes, content_type: str) -> None:
        self._send_bytes(raw, content_type, HTTPStatus.OK)

    def _parse_byte_range(self, range_header: str, total_size: int) -> Tuple[int, int]:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
        if match is None or total_size <= 0:
            raise ValueError("Invalid range")
        start_text, end_text = match.groups()
        if start_text:
            start = int(start_text)
            end = int(end_text) if end_text else total_size - 1
        elif end_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise ValueError("Invalid range")
            start = max(total_size - suffix_length, 0)
            end = total_size - 1
        else:
            raise ValueError("Invalid range")
        if start >= total_size:
            raise ValueError("Range starts past EOF")
        end = min(end, total_size - 1)
        if end < start:
            raise ValueError("Invalid range")
        return start, end

    def _write_range_not_satisfiable(self, total_size: int) -> None:
        try:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", "bytes */{}".format(total_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _send_file_bytes(
        self,
        file_path: Path,
        content_type: str,
        start: int,
        end: int,
        total_size: int,
        status: HTTPStatus,
    ) -> None:
        content_length = 0 if total_size == 0 else max(end - start + 1, 0)
        try:
            with file_path.open("rb") as handle:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(content_length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                if status == HTTPStatus.PARTIAL_CONTENT:
                    self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, total_size))
                self.end_headers()
                if content_length == 0:
                    return
                handle.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            self._write_json({"error": "Failed to read file: {}".format(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _read_json_body(self) -> Dict[str, object]:
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid JSON body: {}".format(exc)) from exc

    def _write_json(self, payload: Dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, indent=2).encode("utf-8")
        self._send_bytes(raw, "application/json; charset=utf-8", status)

    def _send_bytes(self, raw: bytes, content_type: str, status: HTTPStatus) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def run_dashboard_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    server = DashboardHTTPServer((host, port), DashboardRequestHandler, root_dir=root_dir)
    print("[INFO] Dashboard available at http://{}:{}/".format(host, port))
    print("[INFO] Project root: {}".format(root_dir))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down dashboard server.")
    finally:
        server.server_close()


def _resolve_youtube_playable_url(
    youtube_url: str,
    configured_cookie_path: str = "",
    configured_extractor_args: str = "",
) -> str:
    yt_dlp_cmd = find_yt_dlp_command()
    if not yt_dlp_cmd:
        raise RuntimeError("yt-dlp is not installed for this Python and was not found on PATH.")
    yt_dlp_warning = get_yt_dlp_command_warning(yt_dlp_cmd)
    if yt_dlp_warning:
        last_error = yt_dlp_warning
    else:
        last_error = None

    attempts, _attempt_notes = build_playable_url_attempts(
        [Path.cwd(), Path(__file__).resolve().parent.parent],
        configured_cookie_path=configured_cookie_path,
        configured_extractor_args=configured_extractor_args,
    )
    for attempt in attempts:
        try:
            completed = subprocess.run(
                [*yt_dlp_cmd, "--no-playlist", *yt_dlp_network_backoff_args(), *attempt["args"], youtube_url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30.0,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            last_error = str(exc)
            continue
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                candidate = line.strip()
                if candidate.startswith("http://") or candidate.startswith("https://"):
                    return candidate
        stderr = completed.stderr.strip() or completed.stdout.strip()
        if stderr:
            last_error = stderr.splitlines()[-1]
        if output_indicates_rate_limit(stderr.splitlines()):
            raise RuntimeError(
                "Could not resolve a browser-playable video source from YouTube. {} {}".format(
                    last_error or "",
                    rate_limit_help_message(),
                ).strip()
            )
    raise RuntimeError(
        "Could not resolve a browser-playable video source from YouTube. {} "
        "YouTube may be rate-limiting this machine, or yt-dlp may need an update. {}".format(
            last_error or "",
            auth_help_message(),
        )
    )
