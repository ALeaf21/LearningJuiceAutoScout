import os
import platform
import shutil
import subprocess
from typing import Dict, List, Optional


def collect_hardware_profile() -> Dict[str, object]:
    system = platform.system()
    logical_cores = os.cpu_count() or 1
    physical_cores = _detect_physical_cores(system) or logical_cores
    memory_bytes = _detect_memory_bytes(system)
    memory_gb = round(memory_bytes / (1024 ** 3), 2) if memory_bytes else None
    cpu_name = _detect_cpu_name(system)
    gpu_name = _detect_gpu_name(system)

    recommendations = _build_recommendations(
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        memory_gb=memory_gb,
        gpu_name=gpu_name,
    )
    return {
        "system": system,
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_name": cpu_name,
        "logical_cores": logical_cores,
        "physical_cores": physical_cores,
        "memory_bytes": memory_bytes,
        "memory_gb": memory_gb,
        "gpu_name": gpu_name,
        "recommendations": recommendations,
    }


def collect_runtime_usage() -> Dict[str, object]:
    system = platform.system()
    logical_cores = os.cpu_count() or 1
    load_avg = None
    cpu_percent_estimate = None
    try:
        loads = os.getloadavg()
        load_avg = {
            "one_min": round(loads[0], 3),
            "five_min": round(loads[1], 3),
            "fifteen_min": round(loads[2], 3),
        }
        cpu_percent_estimate = round(min(100.0, (loads[0] / max(logical_cores, 1)) * 100.0), 1)
    except (AttributeError, OSError):
        pass

    memory = _detect_memory_usage(system)
    network = _detect_network_usage(system)
    return {
        "system": system,
        "logical_cores": logical_cores,
        "cpu_percent_estimate": cpu_percent_estimate,
        "load_average": load_avg,
        "memory": memory,
        "network": network,
    }


def _build_recommendations(logical_cores: int,
                           physical_cores: int,
                           memory_gb: Optional[float],
                           gpu_name: Optional[str]) -> Dict[str, object]:
    scrape_workers = max(4, min(32, logical_cores * 4))
    if memory_gb is None:
        parallel_track_jobs = max(1, min(physical_cores - 1, 2))
    else:
        memory_bound_jobs = max(1, int(memory_gb // 6.0))
        parallel_track_jobs = max(1, min(max(physical_cores - 1, 1), memory_bound_jobs))
    native_threads = max(1, physical_cores // parallel_track_jobs)

    notes: List[str] = [
        "Use I/O-heavy thread pools for FTC Events scraping and video download discovery.",
        "Keep full tracking jobs process-based rather than thread-based because OpenCV and NumPy work is CPU-heavy.",
        "If you batch multiple matches, cap concurrent tracking jobs near the recommended value to avoid cache and memory contention.",
    ]
    if gpu_name:
        notes.append("Detected GPU: {}. The current tracker is CPU-oriented, so GPU acceleration is mostly useful for video decode or future model-based pipelines.".format(gpu_name))
    if memory_gb is not None and memory_gb < 16:
        notes.append("System memory is modest. Prefer one full tracking job at a time when also writing debug video.")
    elif memory_gb is not None and memory_gb >= 32:
        notes.append("This machine has enough memory for several parallel batch jobs as long as video I/O stays local.")

    return {
        "recommended_scrape_workers": scrape_workers,
        "recommended_parallel_track_jobs": parallel_track_jobs,
        "recommended_native_math_threads_per_job": native_threads,
        "suggested_env": {
            "OMP_NUM_THREADS": str(native_threads),
            "OPENBLAS_NUM_THREADS": str(native_threads),
            "MKL_NUM_THREADS": str(native_threads),
        },
        "notes": notes,
    }


def _detect_physical_cores(system: str) -> Optional[int]:
    if system == "Darwin":
        value = _run_and_capture(["sysctl", "-n", "hw.physicalcpu"])
        return int(value) if value and value.isdigit() else None
    if system == "Linux":
        value = _run_and_capture(["getconf", "_NPROCESSORS_ONLN"])
        return int(value) if value and value.isdigit() else None
    if system == "Windows":
        value = _run_and_capture([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum",
        ])
        return int(value) if value and value.isdigit() else None
    return None


def _detect_memory_bytes(system: str) -> Optional[int]:
    if system == "Darwin":
        value = _run_and_capture(["sysctl", "-n", "hw.memsize"])
        return int(value) if value and value.isdigit() else None
    if system == "Linux":
        value = _run_and_capture(["getconf", "PAGE_SIZE"])
        pages = _run_and_capture(["getconf", "_PHYS_PAGES"])
        if value and pages and value.isdigit() and pages.isdigit():
            return int(value) * int(pages)
        return None
    if system == "Windows":
        value = _run_and_capture([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        return int(value) if value and value.isdigit() else None
    return None


def _detect_cpu_name(system: str) -> Optional[str]:
    if system == "Darwin":
        return _run_and_capture(["sysctl", "-n", "machdep.cpu.brand_string"])
    if system == "Linux":
        value = _run_and_capture(["bash", "-lc", "lscpu | grep 'Model name' | sed 's/.*: *//'"])
        return value or platform.processor() or None
    if system == "Windows":
        return _run_and_capture([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ])
    return platform.processor() or None


def _detect_gpu_name(system: str) -> Optional[str]:
    if system == "Darwin":
        value = _run_and_capture([
            "system_profiler",
            "SPDisplaysDataType",
        ], timeout=8.0)
        if not value:
            return None
        for line in value.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chipset Model:"):
                return stripped.split(":", 1)[1].strip()
        return None
    if shutil.which("nvidia-smi"):
        value = _run_and_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if value:
            return value.splitlines()[0].strip()
    if system == "Linux":
        value = _run_and_capture(["bash", "-lc", "lspci | grep -i 'vga\\|3d\\|display' | head -n 1"])
        return value or None
    if system == "Windows":
        value = _run_and_capture([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)",
        ])
        return value or None
    return None


def _detect_memory_usage(system: str) -> Optional[Dict[str, object]]:
    if system == "Darwin":
        total_raw = _run_and_capture(["sysctl", "-n", "hw.memsize"])
        vm_stat = _run_and_capture(["vm_stat"])
        page_size_raw = _run_and_capture(["sysctl", "-n", "hw.pagesize"])
        if not total_raw or not vm_stat or not page_size_raw:
            return None
        if not total_raw.isdigit() or not page_size_raw.isdigit():
            return None
        total_bytes = int(total_raw)
        page_size = int(page_size_raw)
        pages = {}
        for line in vm_stat.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                pages[key.strip()] = int(digits)
        free_pages = pages.get("Pages free", 0) + pages.get("Pages speculative", 0)
        inactive_pages = pages.get("Pages inactive", 0)
        available_bytes = (free_pages + inactive_pages) * page_size
        used_bytes = max(0, total_bytes - available_bytes)
        used_percent = round((used_bytes / total_bytes) * 100.0, 1) if total_bytes else None
        return {
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "used_percent": used_percent,
        }

    if system == "Linux":
        meminfo = _run_and_capture(["bash", "-lc", "cat /proc/meminfo"])
        if not meminfo:
            return None
        values = {}
        for line in meminfo.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                values[key.strip()] = int(digits) * 1024
        total_bytes = values.get("MemTotal")
        available_bytes = values.get("MemAvailable")
        if not total_bytes or available_bytes is None:
            return None
        used_bytes = max(0, total_bytes - available_bytes)
        used_percent = round((used_bytes / total_bytes) * 100.0, 1)
        return {
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "used_percent": used_percent,
        }

    return None


def _detect_network_usage(system: str) -> Optional[Dict[str, object]]:
    if system in ("Darwin", "Linux"):
        output = _run_and_capture(["netstat", "-ib"], timeout=5.0)
        if not output:
            return None
        total_in = 0
        total_out = 0
        for line in output.splitlines()[1:]:
            parts = line.split()
            if not parts:
                continue
            iface = parts[0]
            if iface.startswith("lo"):
                continue
            numbers = [part for part in parts if part.isdigit()]
            if len(numbers) >= 6:
                total_in += int(numbers[3])
                total_out += int(numbers[6]) if len(numbers) > 6 else 0
        if total_in == 0 and total_out == 0:
            return None
        return {
            "bytes_in_total": total_in,
            "bytes_out_total": total_out,
        }

    return None


def _run_and_capture(command: List[str], timeout: float = 4.0) -> Optional[str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    text = completed.stdout.strip()
    return text or None
