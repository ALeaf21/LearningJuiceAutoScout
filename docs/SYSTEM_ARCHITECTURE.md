# LearningJuiceAutoScout: Complete System Architecture

## System Overview

AutoScout is a computer vision pipeline that automatically tracks four robots in FTC (FIRST Tech Challenge) match videos, detects shots, and exports structured data for analysis.

```mermaid
graph TB
    Input["📹 Input<br/>Video File or YouTube URL"]
    Calibration["🔧 Field Calibration<br/>Auto-detect or Manual"]
    Tracking["🤖 Robot Tracking Pipeline<br/>Background Subtraction + Hungarian Assignment"]
    ShotDetection["🎯 Shot Detection<br/>Ball Tracking + Goal Analysis"]
    Output["📊 Output Formats<br/>CSV / JLOG / WPILog"]
    Dashboard["📈 Dashboard<br/>Real-time Monitoring"]
    
    Input --> Calibration
    Calibration --> Tracking
    Tracking --> ShotDetection
    ShotDetection --> Output
    Tracking -.->|Live Preview| Dashboard
    ShotDetection -.->|Live Preview| Dashboard
    
    style Input fill:#e1f5ff
    style Output fill:#c8e6c9
    style Dashboard fill:#fff9c4
    style Tracking fill:#f3e5f5
    style ShotDetection fill:#f3e5f5
```

---

## Module Organization

### Core Tracking Package (`autoscout/`)

```mermaid
graph LR
    Main["auto_scout.py<br/>(Main Entry)"]
    
    subgraph Core["Core Tracking"]
        Tracker["tracker.py<br/>(RobotTracker)"]
        Shot["shot.py<br/>(ShotDetector)"]
        Models["models.py<br/>(Data Structures)"]
    end
    
    subgraph Utilities["Utilities"]
        Geometry["geometry.py<br/>(Coordinate Systems)"]
        Helpers["helpers.py<br/>(Field Detection)"]
        Runtime["runtime.py<br/>(Console Output)"]
    end
    
    subgraph Export["Output Writers"]
        WPILog["wpilog.py<br/>(WPILog Binary)"]
        JuiceLog["juice_log.py<br/>(JLOG Format)"]
    end
    
    subgraph Integration["External Integration"]
        Hardware["hardware.py"]
        FTC["ftc_events.py"]
        YTDlp["ytdlp.py"]
        Dashboard["dashboard_server.py"]
    end
    
    Main --> Tracker
    Main --> Shot
    Main --> JuiceLog
    Main --> WPILog
    
    Tracker --> Models
    Tracker --> Geometry
    Tracker --> Helpers
    
    Shot --> Models
    Shot --> Helpers
    
    Main -.-> Dashboard
    Main -.-> Hardware
    
    style Main fill:#ffd54f
    style Core fill:#c8e6c9
    style Export fill:#b3e5fc
    style Utilities fill:#ffe0b2
    style Integration fill:#d7ccc8
```

### Data Flow Through Pipeline

```mermaid
sequenceDiagram
    participant Video as Video File
    participant Calib as Field Calibration
    participant Track as Robot Tracking
    participant Shot as Shot Detection
    participant CSV as CSV Export
    participant JLOG as JLOG Export
    participant WPI as WPILog Export
    
    Video->>Calib: Frame Sample
    Calib->>Track: Homography Matrix
    
    Video->>Track: Frame Stream
    Track->>Track: Background Subtraction
    Track->>Track: Blob Detection
    Track->>Track: Hungarian Assignment
    Track->>Track: Update Poses
    
    Track->>Shot: Robot Poses
    Shot->>Shot: Ball Detection
    Shot->>Shot: Launch Detection
    Shot->>Shot: Goal Entry Tracking
    
    Track->>CSV: Poses + Shots
    Track->>JLOG: Poses + Shots
    Track->>WPI: Poses + Shots
    Shot->>CSV: Shot Events
    Shot->>JLOG: Shot Events
    Shot->>WPI: Shot Events
```

---

## Coordinate System Transformations

The system uses three coordinate frames with transformations between them:

```mermaid
graph LR
    User["User Input<br/>(Center-Origin)<br/>(-72, -72) to (72, 72) inches"]
    
    Internal["Internal Tracker<br/>(Corner-Origin)<br/>(0, 0) to (144, 144) inches"]
    
    Output["Output Export<br/>(Center-Origin)<br/>(-72, -72) to (72, 72) inches"]
    
    WPILog["WPILog Format<br/>(Axes Swapped)<br/>Meters"]
    
    User -->|+72| Internal
    Internal -->|-72| Output
    Output -->|*0.0254<br/>Swap Axes| WPILog
    
    style User fill:#e3f2fd
    style Internal fill:#f3e5f5
    style Output fill:#c8e6c9
    style WPILog fill:#fff3e0
```

**Transformation Functions:**
- `_field_center_to_corner_xy(x, y)`: User input → Tracker (add 72)
- `_field_corner_to_center_xy(x, y)`: Tracker → Output (subtract 72)
- `_normalize_angle_rad(angle)`: Heading to (-π, π] range

---

## Robot Tracking Algorithm

```mermaid
graph TD
    Frame["Read Video Frame"]
    FG["Background Subtraction<br/>|frame - median_bg| > threshold"]
    Blobs["Extract Foreground Blobs<br/>Morphological Operations + Contours"]
    
    Init{Initialized?}
    
    Bootstrap["Bootstrap Initialization<br/>Select best 4-blob lineup"]
    Assign["Optimal Assignment<br/>Hungarian Algorithm"]
    
    UpdateMerge["Update Merge Groups<br/>Track Blob Overlaps"]
    UpdatePose["Update Robot Poses<br/>Motion + Heading"]
    
    Output["Emit RobotPose Array<br/>x, y, heading, visible"]
    
    Frame --> FG
    FG --> Blobs
    Blobs --> Init
    
    Init -->|No| Bootstrap
    Init -->|Yes| Assign
    
    Bootstrap --> UpdateMerge
    Assign --> UpdateMerge
    UpdateMerge --> UpdatePose
    UpdatePose --> Output
    
    style Frame fill:#e1f5ff
    style Output fill:#c8e6c9
    style Assign fill:#f3e5f5
    style UpdateMerge fill:#f3e5f5
```

**Key Parameters:**
- `FG_THRESH = 30`: Foreground difference threshold
- `BLOB_MIN = 350`: Minimum blob area (pixels²)
- `MAX_COAST = 60`: Frames track survives without detection
- `REID_COST_WEIGHT = 1.60`: Appearance cost weight

### Hungarian Assignment Cost Matrix

```
Rows: 4 robot tracks
Cols: N detected blobs + 4 skip options

Cost Components:
├─ Motion prediction error (field space)
├─ Motion prediction error (image space)
├─ Blob quality penalty
├─ Appearance distance (re-ID histogram)
├─ Post-merge lock penalties
└─ Coast/reacquisition penalties
```

---

## Merge Group Management (3+ Robot Collisions)

When multiple robots share a foreground blob (collision):

```mermaid
graph TD
    Merge["Merge Started<br/>2+ tracks → 1 blob"]
    
    subgraph Tracking["Per-Frame Updates"]
        Peaks["Extract Peaks<br/>Distance Transform"]
        Assign["Assign Peaks → Tracks<br/>Nearest Neighbor"]
        Vote["Record Permutation<br/>Vote on Identity"]
    end
    
    Separate["Merge Ends<br/>Blob Separates"]
    
    Resolve["Resolve Permutation<br/>Entry vs Current Order"]
    Reanchor["Re-anchor Tracks<br/>Apply Swaps"]
    
    Merge --> Tracking
    Tracking --> Separate
    Separate --> Resolve
    Resolve --> Reanchor
    
    style Merge fill:#ffccbc
    style Separate fill:#ffccbc
    style Resolve fill:#c8e6c9
    style Tracking fill:#f3e5f5
```

**Data Structures:**
- `entry_order`: Track IDs at merge start (immutable)
- `current_order`: Track IDs in current frame (updated per frame)
- `peak_assignment`: Pixel positions of each track within blob
- `order_votes`: Voting matrix for permutation resolution

---

## Shot Detection Pipeline

```mermaid
graph TD
    Frame["Video Frame"]
    BallDet["Ball Detection<br/>HSV Color Masking<br/>Green + Purple"]
    Contours["Extract Contours<br/>Filter by Size<br/>8-420 px²"]
    
    Tracks["Ball Track Association<br/>Nearest Neighbor Search"]
    
    LaunchCheck{"Launch<br/>Velocity OK?"}
    
    GoalCheck["Goal Entry Detection<br/>Polygon-based"]
    
    Resolve{"Track Lost<br/>or<br/>Timeout?"}
    
    MadeOrMissed{"Entered<br/>Goal?"}
    
    Event["Emit ShotEvent<br/>made/missed"]
    
    Frame --> BallDet
    BallDet --> Contours
    Contours --> Tracks
    Tracks --> LaunchCheck
    
    LaunchCheck -->|No| GoalCheck
    LaunchCheck -->|Yes| GoalCheck
    
    GoalCheck --> Resolve
    Resolve -->|No| GoalCheck
    Resolve -->|Yes| MadeOrMissed
    
    MadeOrMissed -->|Yes| Event
    MadeOrMissed -->|No| Event
    
    style Event fill:#c8e6c9
    style BallDet fill:#b2dfdb
    style LaunchCheck fill:#fff59d
    style MadeOrMissed fill:#fff59d
```

**Shot Launch Criteria:**
- Minimum displacement: 8 pixels
- Minimum speed: 2 px/frame
- Minimum upward motion: 3 pixels (y decreases)
- Positive acceleration: ≥35% (speed increasing)
- Life: ≥2 frames

---

## Output Formats

```mermaid
graph LR
    Tracking["Tracking Output<br/>RobotPose Array"]
    Shots["Shot Events<br/>ShotEvent Array"]
    
    CSV["CSV<br/>(robot_positions.csv)<br/>48 columns<br/>Human-readable"]
    
    JLOG["JLOG<br/>(robot_positions.jlog)<br/>Variable-length encoding<br/>90% smaller"]
    
    WPILOG["WPILog<br/>(match_log.wpilog)<br/>AdvantageScope format<br/>Playable in dashboards"]
    
    DEBUG["Debug Outputs<br/>Background Image<br/>Debug Frames/Video"]
    
    Tracking --> CSV
    Tracking --> JLOG
    Tracking --> WPILOG
    Shots --> CSV
    Shots --> JLOG
    Shots --> WPILOG
    Tracking --> DEBUG
    
    style CSV fill:#c8e6c9
    style JLOG fill:#c8e6c9
    style WPILOG fill:#c8e6c9
    style DEBUG fill:#fff9c4
```

### CSV Structure (48 Columns)

```
timestamp_s,
robot0_x_in, robot0_y_in, robot0_heading_rad, robot0_visible,
robot1_x_in, robot1_y_in, robot1_heading_rad, robot1_visible,
robot2_x_in, robot2_y_in, robot2_heading_rad, robot2_visible,
robot3_x_in, robot3_y_in, robot3_heading_rad, robot3_visible,
robot0_shot_result, robot0_shot_x_in, robot0_shot_y_in, robot0_shot_goal,
... (repeated for robots 1-3)
```

**Coordinates:** Center-origin system (0,0 = field center)

---

## Entry Points and CLI

```mermaid
graph LR
    auto_scout["auto_scout.py<br/>(Main Tracker)"]
    calibrate["tools/calibrate.py<br/>(Field Corner Picker)"]
    dashboard["dashboard_server.py<br/>(Web Dashboard)"]
    debug["tools/debug.py<br/>(WPILog Inspector)"]
    
    auto_scout -->|Input| Video["Video File<br/>or YouTube URL"]
    calibrate -->|Input| Video
    
    auto_scout -->|Flags| Flags["--corners<br/>--robot-init-positions<br/>--manual-reference-csv<br/>--debug<br/>--sample-rate"]
    
    auto_scout -->|Output| Output["CSV / JLOG / WPILog"]
    
    dashboard -->|Serve| Web["http://localhost:8765/"]
    
    debug -->|Read| WPILog["match_log.wpilog"]
    
    style auto_scout fill:#ffd54f
    style calibrate fill:#b3e5fc
    style dashboard fill:#fff9c4
    style debug fill:#c8e6c9
```

---

## Performance Characteristics

```mermaid
graph TB
    Input["640×360 @ 30 fps<br/>5-minute match<br/>9000 frames"]
    
    Pipeline["Processing Pipeline"]
    
    Time["Processing Time<br/>~100-200ms/frame<br/>5-10 fps effective"]
    
    Memory["Memory Usage<br/>~5-10 MB<br/>Background: 2.8 MB<br/>State: ~2 KB per track"]
    
    Bottleneck["Bottleneck: Hungarian<br/>Assignment O(n³)<br/>+ Morphological Ops"]
    
    Input --> Pipeline
    Pipeline --> Time
    Pipeline --> Memory
    Time --> Bottleneck
    
    style Input fill:#e1f5ff
    style Bottleneck fill:#ffccbc
    style Time fill:#c8e6c9
    style Memory fill:#c8e6c9
```

---

## Error Handling & Recovery

```mermaid
graph TD
    Issue["Issue Detected"]
    
    NoField["Field Corners<br/>Not Detected"]
    RobotLost["Robot Lost<br/>During Occlusion"]
    MergeAmbig["Merge Permutation<br/>Ambiguous"]
    
    Resolve1["→ Use calibrate.py<br/>or --corners flag"]
    Resolve2["→ Re-acquisition window<br/>or manual re-ID CSV"]
    Resolve3["→ Apply best-guess<br/>permutation + logging"]
    
    Issue --> NoField
    Issue --> RobotLost
    Issue --> MergeAmbig
    
    NoField --> Resolve1
    RobotLost --> Resolve2
    MergeAmbig --> Resolve3
    
    style Issue fill:#ffcdd2
    style NoField fill:#ffccbc
    style RobotLost fill:#ffccbc
    style MergeAmbig fill:#ffccbc
    style Resolve1 fill:#c8e6c9
    style Resolve2 fill:#c8e6c9
    style Resolve3 fill:#c8e6c9
```

---

## Integration with FIRST Ecosystem

```mermaid
graph TB
    AutoScout["AutoScout<br/>Tracking Pipeline"]
    
    WPILog["WPILog Export"]
    AdvantageScope["AdvantageScope<br/>Post-Match Analysis"]
    
    FTCEvents["FTC Events API<br/>Match Metadata"]
    YouTube["YouTube<br/>Official Broadcasts"]
    
    CSV["CSV Export"]
    Custom["Custom Analysis<br/>Jupyter Notebooks"]
    
    AutoScout --> WPILog
    WPILog --> AdvantageScope
    
    AutoScout --> FTCEvents
    AutoScout --> YouTube
    
    AutoScout --> CSV
    CSV --> Custom
    
    style AutoScout fill:#ffd54f
    style WPILog fill:#b3e5fc
    style CSV fill:#c8e6c9
    style AdvantageScope fill:#fff9c4
    style FTCEvents fill:#d7ccc8
    style Custom fill:#ffe0b2
```

---

## Key Design Patterns

### 1. State Machine (Tracker Lifecycle)

```
[Uninitialized]
    ↓ (detect 4 valid blobs)
[Initialized]
    ↓ (loop until video ends)
    ├─ Extract blobs
    ├─ Assign to tracks (Hungarian)
    ├─ Manage merges
    ├─ Emit poses
    └─ (repeat)
```

### 2. Cost-Matrix Assignment

Minimize total cost subject to: each track ≤ 1 blob, each blob ≤ 1 track

**Cost = Distance + Quality + Appearance + Penalties**

### 3. Homography-Based Perspective Transform

```
Image Space (pixels)  ←→  Field Space (inches)
    via OpenCV.findHomography(corners, dst2d)
```

### 4. Merge Group Voting

For 3+ robot merges, use voting to resolve permutations:
```
entry_order[0]  →  robot at position 0 at merge start
current_order[0] → robot at position 0 in current frame
Permutation = mapping entry → current
```

---

## Quick Reference: Common Workflows

### Track Local Video with Known Corners

```bash
python3 auto_scout.py \
  --no-download \
  --video-path match.mp4 \
  --corners field_corners.json
```

### Calibrate Field Corners

```bash
python3 tools/calibrate.py match.mp4 --output field_corners.json --frame 100
```

### Enable Debug Output

```bash
python3 auto_scout.py ... --debug --debug-video --debug-every 5
```

### Bootstrap Re-ID Model

```bash
# 1. Create manual_poses.csv with hand-labeled positions
# 2. Run tracker with:
python3 auto_scout.py ... --manual-reference-csv manual_poses.csv
```

### Start Web Dashboard

```bash
python3 dashboard_server.py --host 127.0.0.1 --port 8765
# Open http://127.0.0.1:8765/ in browser
```

---

## Testing & Validation

```mermaid
graph TD
    Test["Test Workflow"]
    
    Metrics["Track Accuracy<br/>Shot Detection Precision<br/>Shot Detection Recall<br/>Merge Resolution Success"]
    
    Videos["Test Videos<br/>Short: 1-2 min<br/>Medium: 5-8 min<br/>Challenging: Merges + Motion"]
    
    Checks["Validation Checks<br/>✓ CSV row count correct<br/>✓ JLOG decodes to CSV<br/>✓ WPILog plays in AdvantageScope<br/>✓ Debug video shows robot outlines<br/>✓ No duplicate shot events"]
    
    Test --> Metrics
    Test --> Videos
    Test --> Checks
    
    style Metrics fill:#c8e6c9
    style Videos fill:#b3e5fc
    style Checks fill:#fff9c4
```

---

## Architecture Summary

**Strengths:**
- Robust tracking via Hungarian assignment + re-ID
- Handles complex merge scenarios with voting system
- Multiple output formats for different use cases
- Real-time dashboard for monitoring

**Components:**
- 11 core tracking modules
- 4+ utility modules
- 3 export formats (CSV, JLOG, WPILog)
- Web dashboard + CLI tools

**Performance:**
- 10 fps on modern 4-core CPU (640×360 video)
- ~5-10 MB memory footprint
- 90% file size reduction with JLOG format

**Integration:**
- AdvantageScope playback via WPILog
- FTC Events discovery
- YouTube integration via yt-dlp
- Custom analysis via CSV export
