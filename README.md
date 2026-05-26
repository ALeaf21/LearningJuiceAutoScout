![Project Header Image](assets/Project-REDACTED-Header.png)

# *FIRST* Tech Challenge Automatic Scouter
| Codename | **Project REDACTED** |
| -------- | ---------------------|


*This project is developed by **Ray Enterprises' R&D** (ARED) for use by FTC Team Juice 16236.*


https://github.com/user-attachments/assets/64045d5d-e12f-4a64-a7b1-fd415e46685a


## About AutoScout
`auto_scout.py` is the tracker entrypoint for FTC match videos. It orchestrates the tracking pipeline, projects robots into field coordinates, and exports the result for later analysis.

Current outputs are focused on robot motion:
- `robot_positions.csv` for per-timestamp robot poses and visibility
- `robot_positions.jlog` for compact binary robot poses and shot events
- `match_log.wpilog` for AdvantageScope playback
- optional annotated debug frames in `output/tracker_debug/`
- optional annotated debug video in `output/tracker_debug.mp4`

The CSV and JLOG pose logs include per-robot shot-event fields:

- `robot#_shot_result`
- `robot#_shot_x_in`
- `robot#_shot_y_in`
- `robot#_shot_goal`

These are populated on the frame where AutoScout resolves a shot as `made` or `missed`.

Output coordinates now use a center-origin field frame:

- `(0, 0)` is the center of the 144 inch x 144 inch field
- `x` increases to the right
- `y` increases downward in the projected field plane used by the tracker
- the field corners are approximately `(-72, -72)` to `(72, 72)` in inches

## Requirements
Install the runtime dependencies:

```bash
python3 -m pip install opencv-python numpy progress
python3 -m pip install scipy
```

`scipy` is technically optional, but the tracker uses it for the best blob-to-track assignment.

If you upgraded Python, reinstall optional tools like `yt-dlp` into the same interpreter you use to launch AutoScout:

```bash
python3 -m pip install -U yt-dlp
```

## Quickstart
Track a local video:

```bash
python3 auto_scout.py --no-download --video-path match.mp4 --corners field_corners.json
```

Track from YouTube:

```bash
python3 auto_scout.py "https://www.youtube.com/watch?v=..."
```

Start the local dashboard:

```bash
python3 dashboard_server.py --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/` in a browser.

The dashboard adds:

- FTC Events event loading by season plus event code
- qualification-match overview cards with alliance metadata, scores when available, and per-match status
- a hardware-aware global queue that can keep starting ready matches automatically
- local hardware inspection, runtime usage, and concurrency recommendations
- background tracker-job launching, stopping, live previews, stage checklists, terminal output, and embedded data visualization
- an in-browser field-corner calibration tool that can use uploaded files, queued YouTube clips, example videos, prior source videos, or still-cached match downloads
- yt-dlp cookies and extractor-args settings for both downloads and browser-playable YouTube resolution

## CLI Flags
| Flag | Type | Default | Description |
| -------- | -------- | -------- | -------- |
| `--output-dir` | `string` | `./output` | Directory for CSV, JLOG, WPILOG, background image, and debug frames |
| `--start-offset` | `float` | `0.0` | Seconds to skip before the match timer starts |
| `--sample-rate` | `float` | `10.0` | Effective frames per second to process |
| `--debug` | - | disabled | Save annotated debug frames to `tracker_debug/` |
| `--debug-video` | - | disabled | Write annotated debug output to `tracker_debug.mp4` instead of `tracker_debug/` images. Implies `--debug` |
| `--debug-every` | `int` | `1` | Save one debug frame every `N` processed frames |
| `--no-download` | - | disabled | Use a local video instead of downloading from YouTube |
| `--video-path` | `string` | `None` | Path to the local match video. Requires `--no-download` |
| `--corners` | `string` | `None` | Path to `field_corners.json` from `tools/calibrate.py` |
| `--robot-init-positions` | `json` | `None` | Four starting field coordinates in inches, using center-origin coordinates |
| `--manual-reference-csv` | `string` | `None` | Manual `robot_positions`-style CSV or JLOG in the same center-origin coordinate system |

## Field Calibration
AutoScout can try to detect the field outline automatically, but manual calibration is more reliable.

Generate field corners from a local video:

```bash
python3 tools/calibrate.py path/to/match.mp4
```

That produces a `field_corners.json` file. Pass it to `auto_scout.py` with `--corners`. Recalibrate whenever the camera angle or crop changes.

## Repository Layout

- `auto_scout.py`
  Thin entrypoint that wires together the tracker pipeline, exports, and CLI.
- `dashboard_server.py`
  Starts the local dashboard web server.
- `autoscout/`
  Main Python package for the tracker internals.
  Includes `tracker.py`, `shot.py`, `runtime.py`, `geometry.py`, `models.py`, `helpers.py`, `wpilog.py`, `ftc_events.py`, `hardware.py`, and `dashboard_server.py`.
- `dashboard_static/`
  No-build dashboard front-end files served by `dashboard_server.py`.
- `util/`
  Shared helpers including `juice_log.py` and `jlog.js` for the compact `robot_positions.jlog` format.
- `tools/`
  Helper scripts and browser UIs: `calibrate.py`, `debug.py`, `manual_tracker.html`, `data_visualizer.html`, and `shot_visualizer.html`.
- `assets/`
  Static project images.
- `examples/`
  Example local videos for testing and manual workflows.
- `output/`
  Default generated outputs such as CSV, JLOG, WPILOG, background images, and debug media.
- `field_corners.json`
  A local calibration file in the repo root. You can also generate other corner JSON files and pass them with `--corners`.
- `README.md`, `DOCUMENTATION.md`, `AUTOSCOUT_PAPER.tex`
  Quickstart, full reference, and the paper-style writeup.

## Outputs
When a run finishes, the output directory typically contains:

- `robot_positions.csv`
  Flat table of timestamped robot positions, headings, visibility flags, and shot events. `x/y` are in inches from field center.
- `robot_positions.jlog`
  Compact binary JUICE LOG output for the same tracker table data, stored in a self-describing schema format with CRC-checked chunks so truncated writes keep earlier completed data readable.
- `match_log.wpilog`
  WPILOG output using center-origin coordinates converted to meters with the viewer-facing axis remap `x' = y`, `y' = x`, and `heading' = π/2 - heading`.
- `median_background.jpg`
  The median background image used for subtraction.
- `tracker_debug/`
  Optional annotated frames showing contours, split centers, merge annotations, and robot IDs.
- `tracker_debug.mp4`
  Optional annotated debug video written when `--debug-video` is enabled.

When a job is launched from the dashboard, outputs are typically placed under:

- `<output_root>/<EVENT_CODE>/<match_slug>/`

where `output_root` defaults to `./output_dashboard`.

Dashboard YouTube jobs may also create a temporary `match_video.mp4` inside that match directory while the tracker is running. If the job completes successfully, the dashboard deletes that cached download after processing. If the job fails or is stopped before completion, the downloaded video is left in place so it can still be inspected, calibrated against, or reused manually.

## Debug Notes
- `--debug-every` is based on processed-frame spacing. The default `1` saves every processed frame.
- `--debug-video` writes only the sampled debug frames into the video, so playback speed is based on the debug sampling cadence rather than real-time source-video timing.
- Merge debug behavior can also be influenced by the top-level `SAVE_ALL_DEBUG_AROUND_MERGES` flag in `auto_scout.py`.
- Each debug run clears old `.jpg` files from `output/tracker_debug/` before writing new ones.

## AdvantageScope
To inspect the resulting WPILOG:

1. Open AdvantageScope.
2. Open `match_log.wpilog`.
3. Add a `2D Field` tab.
4. Set the field to the FTC season field you want.
5. Drag each `Robot#/Pose` entry into the field poses list.

## Contributing
See `CONTRIBUTING.md` for setup, validation, and pull request guidelines.

## License
This project is licensed under the MIT license, please see `LICENSE` for more information.
