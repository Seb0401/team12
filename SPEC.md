# EX-5600 Shovel Productivity Analysis — Technical Spec

## 1. Problem Statement

**Theme**: Mining Productivity 2.0

Using 15 minutes of stereo video + IMU data from a Hitachi EX-5600 hydraulic shovel loading haul trucks, build a system that measures productivity and provides actionable operator recommendations.

**Productivity framework (OEE for mining)**:

```
Productivity (tonnes/hour) = Payload × Cycles/hour × Utilization × Efficiency
```

| Term          | Meaning                                    | Typical Range                    |
|---------------|--------------------------------------------|----------------------------------|
| Payload       | Tonnes moved per bucket                    | EX-5600 bucket ≈ 29 m³ ≈ 52 t   |
| Cycles/hour   | Dig→dump cycles per hour                   | 60–120 depending on conditions   |
| Availability  | % of shift mechanically operational        | 85–92% (out of scope)           |
| Utilization   | % of available time actually digging       | 75–88%                          |
| Efficiency    | Actual output ÷ theoretical best           | The hidden lever                 |

---

## 2. Input Data

| File                  | Format      | Details                                         |
|-----------------------|-------------|--------------------------------------------------|
| `shovel_left.mp4`     | H.264 video | 1280×720, 15 fps, ~10.5 min (≈9400 frames)      |
| `shovel_right.mp4`    | H.264 video | 1280×720, 15 fps, ~10.5 min (stereo pair)        |
| IMU `.npy`            | NumPy array | Shape (9403, 11): timestamp + 3-ax accel + 3-ax gyro + 4 quaternion |

**Camera mount**: On EX-5600 body, looking downward. Visible: bucket, arm joint, boom, ground, dump zone. Truck body NOT visible from this angle.

**IMU columns** (inferred):
| Col | Meaning                |
|-----|------------------------|
| 0   | Timestamp (epoch ns)   |
| 1–3 | Accelerometer (x,y,z)  |
| 4–6 | Gyroscope (x,y,z)      |
| 7–10| Quaternion (w,x,y,z)   |

---

## 3. Detection Strategy

### 3.1 YOLOv8 Fine-Tuned Model

**Classes (3)**:
- `bucket` — The excavator bucket (visible in most frames)
- `arm_joint` — Hydraulic arm joint connecting boom to stick
- `boom` — Main boom structure

**Training plan**:
1. Extract 50–100 diverse frames from left video (different cycle phases)
2. Label on Roboflow/CVAT with bounding boxes for 3 classes
3. Fine-tune YOLOv8n or YOLOv8s (few epochs, CUDA GPU)
4. Commit trained weights to `solution/detection/model/`

**Why these classes**: Joint angle geometry between bucket/arm_joint/boom positions enables kinematic phase classification (dig, swing, dump, return) without needing truck detection.

### 3.2 Inference

- Process **every frame** with YOLOv8 (CUDA, ~2-5ms/frame for YOLOv8n)
- Output: bounding boxes + confidence per class per frame
- Filter: confidence threshold ≥ 0.5

---

## 4. Kinematics — Joint Angle Analysis

### 4.1 Angle Computation

From YOLO detections, compute center points of each bounding box:
- `P_bucket` = center of bucket bbox
- `P_arm` = center of arm_joint bbox
- `P_boom` = center of boom bbox

**Angles**:
- `θ_arm` = angle at arm_joint (bucket—arm_joint—boom)
- `θ_bucket` = angle of bucket relative to arm
- `θ_boom` = angle of boom relative to vertical/horizon

These angles change predictably through cycle phases.

### 4.2 Cycle Phase FSM (Finite State Machine)

```
States: IDLE → DIG → SWING_TO_DUMP → DUMP → SWING_TO_DIG → (repeat)
```

| Phase          | Visual Cues                              | Angle Signature                     | IMU Signature                        |
|----------------|------------------------------------------|--------------------------------------|---------------------------------------|
| DIG            | Bucket at ground, material entering      | θ_arm decreasing (curling)          | High accel vibration, low gyro       |
| SWING_TO_DUMP  | Arm retracting, scene rotating           | θ_boom changing, all joints moving  | High gyro Z (rotation)               |
| DUMP           | Bucket tilted, material falling          | θ_bucket max extension              | Brief accel spike, low gyro          |
| SWING_TO_DIG   | Return swing, empty bucket               | θ_boom changing back                | High gyro Z (opposite direction)     |
| IDLE           | No significant motion                    | Angles stable                       | Low accel, low gyro                  |

**Transition logic**: Threshold-based on angle derivatives + IMU confirmation.

---

## 5. IMU Processing

### 5.1 Signal Processing
- Low-pass filter accelerometer (remove vibration noise)
- Band-pass filter gyroscope (isolate swing frequency ~0.1–2 Hz)
- Compute magnitude: `|gyro| = sqrt(gx² + gy² + gz²)`

### 5.2 IMU-Only Cycle Detection
- **Swing detection**: Peaks in gyro Z-axis magnitude → mark swing events
- **Dig detection**: High-frequency vibration in accelerometer → digging
- **Idle detection**: Low variance in all channels → idle period
- Peak detection: `scipy.signal.find_peaks` with minimum distance between cycles

### 5.3 Fusion
- Align IMU timestamps with video frame indices using epoch timestamps
- Cross-validate: video-detected phase transitions must match IMU events within ±0.5s tolerance
- Conflict resolution: IMU takes priority for timing, video for phase classification

---

## 6. Stereo Vision

### 6.1 Depth Estimation
- **Method**: OpenCV `StereoSGBM` (Semi-Global Block Matching)
- **Sampling**: Every 15th frame (1 depth map/second, ~627 depth maps total)
- **Assume**: Cameras pre-rectified (common for industrial stereo rigs). If not, apply rectification.

### 6.2 Bucket Fill Volume
1. Segment bucket region from YOLO detection bbox
2. Extract depth map within bucket region
3. Fit reference plane (empty bucket baseline)
4. Volume = integral of (depth_material - depth_empty) over bucket area
5. Convert pixels→meters using stereo baseline + focal length

### 6.3 3D Joint Kinematics
1. For each detected joint (bucket, arm_joint, boom):
   - Get pixel coordinates (u, v) from both left and right frames
   - Triangulate → 3D position (X, Y, Z) in world coordinates
2. Compute 3D joint angles (more accurate than 2D projection)
3. Track joint velocities and accelerations over time

---

## 7. Productivity Metrics

### 7.1 Computed Metrics
| Metric                  | Source                           | Unit       |
|-------------------------|----------------------------------|------------|
| Total cycles            | Cycle FSM count                  | count      |
| Average cycle time      | FSM timestamps                   | seconds    |
| Cycles per hour         | Total cycles / duration × 3600   | cycles/hr  |
| Dig time per cycle      | Phase duration                   | seconds    |
| Swing time per cycle    | Phase duration                   | seconds    |
| Dump time per cycle     | Phase duration                   | seconds    |
| Idle time (total)       | IDLE state duration              | seconds    |
| Utilization             | (total - idle) / total × 100     | %          |
| Avg bucket fill         | Stereo volume estimation         | m³         |
| Estimated payload       | Fill volume × material density   | tonnes     |
| Estimated tonnes/hour   | Payload × cycles/hr × util       | t/hr       |

### 7.2 Cycle-by-Cycle Breakdown
For each detected cycle:
- Start/end timestamps
- Phase durations (dig, swing_to_dump, dump, swing_to_dig)
- Bucket fill estimate
- Efficiency score vs best cycle

---

## 8. Output

### 8.1 Annotated Video (`outputs/annotated_video.mp4`)
- Bounding boxes: bucket (green), arm_joint (yellow), boom (blue)
- Current phase label overlay (top-left)
- Cycle counter overlay
- Joint angle display
- Real-time metrics bar (cycle time, fill %)

### 8.2 JSON Report (`outputs/metrics.json`)
```json
{
  "summary": {
    "total_duration_sec": 626.87,
    "total_cycles": 12,
    "avg_cycle_time_sec": 45.2,
    "cycles_per_hour": 79.6,
    "utilization_pct": 85.3,
    "avg_bucket_fill_m3": 24.1,
    "estimated_payload_tonnes": 43.4,
    "estimated_productivity_tph": 3454.2,
    "total_idle_time_sec": 32.5
  },
  "cycles": [
    {
      "cycle_id": 1,
      "start_sec": 5.2,
      "end_sec": 48.7,
      "duration_sec": 43.5,
      "phases": {
        "dig_sec": 12.3,
        "swing_to_dump_sec": 8.1,
        "dump_sec": 4.2,
        "swing_to_dig_sec": 7.8,
        "idle_sec": 11.1
      },
      "bucket_fill_m3": 25.3,
      "efficiency_vs_best": 0.92
    }
  ],
  "recommendations": [
    "Cycle 4 had 18s idle between dump and return swing — investigate truck positioning",
    "Average swing time 8.1s — optimal is <6s, consider swing speed calibration"
  ]
}
```

---

## 9. Architecture

```
solution/
├── main.py                      # Entrypoint: orchestrates full pipeline
├── config.py                    # Paths, constants, thresholds
├── data/
│   ├── __init__.py
│   ├── loader.py                # Load video + IMU, validate inputs
│   └── stereo.py                # Stereo calibration + rectification
├── detection/
│   ├── __init__.py
│   ├── detector.py              # YOLOv8 inference wrapper
│   └── model/                   # Fine-tuned weights (.pt file)
│       └── .gitkeep
├── kinematics/
│   ├── __init__.py
│   ├── joint_angles.py          # Compute angles from YOLO detections
│   ├── cycle_detector.py        # FSM: dig→swing→dump→return
│   └── phase_classifier.py      # Classify phase from angles + IMU
├── imu/
│   ├── __init__.py
│   ├── processor.py             # Signal processing, filtering
│   ├── cycle_detector.py        # IMU-only cycle detection (gyro peaks)
│   └── fusion.py                # Fuse IMU + video cycle detections
├── stereo/
│   ├── __init__.py
│   ├── depth.py                 # Disparity → depth map (sampled)
│   ├── volume.py                # Bucket fill volume estimation
│   └── kinematics_3d.py         # 3D joint positions from stereo
├── productivity/
│   ├── __init__.py
│   ├── metrics.py               # OEE calculation engine
│   └── recommendations.py       # Generate operator recommendations
├── output/
│   ├── __init__.py
│   ├── annotator.py             # Draw bboxes + overlays on video
│   └── report.py                # Generate JSON metrics report
└── utils/
    ├── __init__.py
    ├── time_sync.py             # Sync video frames ↔ IMU timestamps
    └── visualization.py         # Debug plots (matplotlib)
```

---

## 10. Pipeline Flow

```
┌─────────────────────────────────────────────────────┐
│                  main.py (orchestrator)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Load data                                       │
│     ├── loader.load_videos(left, right)             │
│     └── loader.load_imu(npy_path)                   │
│                                                     │
│  2. Process IMU (parallel-ready)                    │
│     ├── processor.filter_signals(imu_data)          │
│     └── imu_cycle_detector.detect_cycles(filtered)  │
│                                                     │
│  3. Process video (frame loop)                      │
│     for frame_idx in range(total_frames):           │
│     │  ├── detector.detect(frame_left)              │
│     │  ├── joint_angles.compute(detections)         │
│     │  ├── if frame_idx % 15 == 0:                  │
│     │  │     ├── depth.compute(left, right)         │
│     │  │     ├── volume.estimate(depth, bucket_bb)  │
│     │  │     └── kin3d.triangulate(left, right)     │
│     │  ├── cycle_detector.update(angles, imu_state) │
│     │  └── annotator.draw(frame, detections, phase) │
│     │                                               │
│  4. Fuse cycles                                     │
│     └── fusion.merge(video_cycles, imu_cycles)      │
│                                                     │
│  5. Compute productivity                            │
│     ├── metrics.compute_oee(fused_cycles, volumes)  │
│     └── recommendations.generate(metrics)           │
│                                                     │
│  6. Output                                          │
│     ├── annotator.finalize_video()                  │
│     └── report.write_json(metrics, cycles)          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 11. Performance Budget

Target: < 10 minutes on standard laptop (with CUDA).

| Step                     | Est. Time        | Notes                            |
|--------------------------|------------------|----------------------------------|
| Load data                | ~2s              | Video opened lazily              |
| IMU processing           | ~1s              | 9400 rows, numpy ops             |
| YOLO inference (9400 fr) | ~20-50s (GPU)    | YOLOv8n: ~2-5ms/frame on CUDA   |
| Stereo depth (627 maps)  | ~60-120s         | StereoSGBM at 1280×720          |
| Kinematics + FSM         | ~5s              | Pure computation                 |
| Annotation + write video | ~30-60s          | OpenCV VideoWriter               |
| JSON report              | <1s              | Serialization                    |
| **Total estimate**       | **~2-4 min**     | Well under 10 min budget         |

---

## 12. Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
ultralytics>=8.0.0
torch>=2.0.0
```

---

## 13. Risks & Mitigations

| Risk                                        | Mitigation                                              |
|---------------------------------------------|---------------------------------------------------------|
| YOLO fine-tune insufficient with 50 frames  | Augmentation (flip, rotate, brightness), fallback to heuristics |
| Stereo cameras not calibrated/rectified     | Check first frame pair, apply rectification if needed   |
| Joint angles unreliable in 2D projection    | Stereo 3D triangulation as primary, 2D as fallback      |
| IMU timestamp drift vs video               | Cross-correlate swing events for alignment offset        |
| Dust/debris occluding detections            | Temporal smoothing, confidence filtering, interpolation  |
| Test dataset different scene/conditions      | Parameterize thresholds, avoid overfitting to dev data   |

---

## 14. Manual Prerequisites

Before running the pipeline:
1. **Label frames**: Extract 50-100 frames → label on Roboflow/CVAT → export YOLOv8 format
2. **Train model**: `yolo train data=dataset.yaml model=yolov8n.pt epochs=50`
3. **Place weights**: Copy `best.pt` to `solution/detection/model/`
