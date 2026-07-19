# LearningJuiceAutoScout: High-Level Architecture

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
