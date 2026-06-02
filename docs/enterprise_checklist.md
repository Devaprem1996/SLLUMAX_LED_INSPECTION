# Enterprise-Level LED Pin Inspection Deployment Checklist

This checklist details the steps required to transition the LED Pin Inspection System from its current local prototype state to a continuous, high-availability production environment on the factory floor.

---

## 1. Hardware, Optics & Enclosure Setup

> [!IMPORTANT]
> The reliability of computer vision (CV) is directly proportional to the consistency of the physical environment. Most vision system failures are due to lighting shifts or camera movement.

- [ ] **Controlled Illumination Enclosure**:
  - [ ] Construct a matte-black light-shielding dome or tunnel over the conveyor inspection zone to block ambient factory light.
  - [ ] Install a high-frequency LED ring-light or coaxial light (powered by a stabilized DC regulator) to ensure constant contrast and prevent camera-shutter flicker.
- [ ] **Industrial Grade Camera & Lens**:
  - [ ] Replace consumer-grade USB webcams with an industrial GigE Vision or USB3 Vision camera (e.g., Basler, FLIR, or IDS).
  - [ ] Fit a low-distortion C-mount lens with locking screws for both focus and aperture to prevent vibration-induced drift.
- [ ] **Rigid Mounting**:
  - [ ] Securely mount the camera on an extrusion frame physically decoupled from the vibrating conveyor motor.
- [ ] **Air-Purged Enclosure**:
  - [ ] Enclose the camera in an IP65/IP67 rated housing with an air-nozzle purge if the factory environment is dusty or oil-heavy.

---

## 2. Real-Time Video & Streaming Pipeline

- [ ] **Industrial Protocol Streaming**:
  - [ ] Stream video using RTSP over Ethernet or native camera SDKs (e.g., PyPylon, GenICam) rather than basic OS-level webcam drivers.
- [ ] **Hardware-Accelerated Decoding**:
  - [ ] Configure OpenCV with a GStreamer backend to utilize GPU/VPU hardware decoding (e.g., NVIDIA NVDEC or Intel VAAPI) to minimize CPU overhead.
- [x] **Frame-Dropping Policy**:
  - [x] Implement a producer-consumer thread queue: if the inspection engine experiences a temporary spike, old frames should be dropped rather than causing inspection lag (buffering queue build-up).

---

## 3. Industrial Control & PLC Integration

> [!IMPORTANT]
> The inspection system must interface directly with the assembly line machinery to act on defects.

- [ ] **Conveyor Reject Hardware**:
  - [ ] Install a physical pneumatic pusher, air-jet, or divert-gate downstream of the camera to eject failed parts.
- [x] **PLC Communication Protocol**:
  - [x] Integrate an industrial communication driver:
    - **OPC-UA**: For enterprise MES integration.
    - **Modbus TCP**: For simple direct PLC read/write registers.
    - **Direct Digital I/O**: Via a USB relay module connected to the PC, sending a 24V pulse to trigger the reject mechanism. (Simulated in production build)
- [ ] **Conveyor Synchronization**:
  - [ ] Connect a photoelectric sensor (part-present sensor) to the PLC. The PLC triggers the camera stream capturing state when the part is physically aligned, removing reliance on motion differencing.

---

## 4. Configuration & Runtime Management

- [x] **Externalized Configuration (YAML/JSON)**:
  - [x] Move all configuration variables (conveyor settle time, model paths, sub-pixel shift limits, calibration offsets) to a production configuration file (e.g., `config.json`).
- [ ] **Dynamic Calibration (Home Position Align)**:
  - [ ] Provide a locked-down "Calibration Mode" UI screen where a technician can place a reference master connector, recalculate the default `PINS` coordinates grid, and save them.
- [x] **Multi-Product Profile Management**:
  - [x] Support loading different configuration profiles for different connector families (e.g., 22-pin, 18-pin, or different pitch sizes).

---

## 5. Storage, Data Pruning & MES Logging

- [x] **Production Logging Database**:
  - [x] Store inspection logs in a local SQLite database or send them to an enterprise time-series database (e.g., InfluxDB / PostgreSQL) rather than a flat JSON file.
- [x] **Pruning Policy for Snapshots**:
  - [x] Set up an automatic disk-space cleanup daemon:
    - Save all `FAIL` images for forensic analysis.
    - Save only a small percentage of `PASS` images (or auto-delete them after 7 days) to prevent running out of SSD storage.
- [ ] **MES Integration**:
  - [ ] Feed pass/fail counts to the factory Manufacturing Execution System (MES) to track yield metrics per shift.

---

## 6. Edge AI Model Roadmap (Hybrid Upgrade)

- [ ] **YOLOv8 Edge Deployment**:
  - [ ] If product shapes vary heavily, train a lightweight object detection model (e.g., YOLOv8-OBB - Oriented Bounding Boxes) to find pin tips.
- [ ] **ONNX Runtime Export**:
  - [ ] Export the trained model to `.onnx` or Intel OpenVINO formats to run highly optimized on local CPUs or cheap edge accelerators (like Raspberry Pi 5 or NVIDIA Jetson Nano).

---

## 7. Operator Interface & Factory HUD

- [x] **Local Operator Screen (HMI)**:
  - [x] Build a simple, full-screen touch GUI (using PyQt or a web panel) showing:
    - Large green **PASS** / red **FAIL** block.
    - Live feed showing aligned housing.
    - Yield rates (Total parts scanned, Pass Rate %, Defect Types count).
- [ ] **Physical Control Buttons**:
  - [ ] Map physical pushbuttons (e.g., "Manual Reset / Retest" and "Mute Alarm") to keyboard events or PLC register triggers.
- [ ] **Software Watchdog**:
  - [ ] Run the Python script inside a system daemon (systemd on Linux, or Windows Service) configured to automatically restart if the camera crashes or the script exits unexpectedly.
