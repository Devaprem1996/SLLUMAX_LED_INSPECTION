import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add parent directory (workspace root) to python path to allow src.core.* imports
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

import cv2
import numpy as np
import time
import json
import sqlite3
import threading
import math
import urllib.parse
import http.server
import socketserver
from urllib.parse import parse_qs, urlparse

from src.core.inspector import align_frame, warp_with_homography, inspect_frame, draw_hud

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'results'))
DB_DIR = os.path.join(RESULTS_DIR, 'db_snapshots')
DB_PATH = os.path.join(RESULTS_DIR, 'inspection.db')
DASHBOARD_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, 'dashboard'))

# Ensure database snapshot directory exists
os.makedirs(DB_DIR, exist_ok=True)

def connect_db(path):
    """Establishes database connection configured with WAL and busy timeout for thread safety."""
    conn = sqlite3.connect(path, timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception as e:
        print(f"Warning: Failed to set PRAGMA parameters: {e}")
    return conn

def validate_alignment(H, stats):
    """Verifies that SIFT homography alignment results match a physically possible connector position."""
    if H is None or not stats:
        return False
    scale = stats.get("scale", 1.0)
    rotation = stats.get("rotation", 0.0)
    inliers = stats.get("inliers", 0)
    
    # Allowed geometry ranges for physical connector in fixture
    scale_ok = 0.80 <= scale <= 1.20
    rotation_ok = -25.0 <= rotation <= 25.0
    inliers_ok = inliers >= 10
    
    return scale_ok and rotation_ok and inliers_ok

def init_db():
    """Initializes SQLite database schema for enterprise logging."""
    conn = connect_db(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Create inspection history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspection_history (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            overall_status TEXT,
            passed_pins INTEGER,
            warning_pins INTEGER,
            severe_bent_pins INTEGER,
            missing_pins INTEGER,
            snapshot_file TEXT,
            tx REAL,
            ty REAL,
            rotation REAL,
            scale REAL,
            inliers INTEGER
        )
    """)
    
    # 2. Create pin metrics detail table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pin_detail_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id TEXT,
            pin_id TEXT,
            status TEXT,
            ncc REAL,
            shift_px REAL,
            ratio REAL,
            detail TEXT,
            FOREIGN KEY(inspection_id) REFERENCES inspection_history(id)
        )
    """)
    
    conn.commit()
    conn.close()

def log_to_sqlite(data):
    """Logs inspection and detailed pin metrics to local SQLite database."""
    conn = connect_db(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Insert main inspection log
        cursor.execute("""
            INSERT OR REPLACE INTO inspection_history 
            (id, timestamp, overall_status, passed_pins, warning_pins, severe_bent_pins, missing_pins, snapshot_file, tx, ty, rotation, scale, inliers)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["item_id"],
            data["timestamp"],
            data["overall_status"],
            data["metrics"]["passed_pins"],
            data["metrics"]["warning_pins"],
            data["metrics"]["severe_bent_pins"],
            data["metrics"]["missing_pins"],
            data["snapshot_file"],
            data["sift_alignment"]["tx"],
            data["sift_alignment"]["ty"],
            data["sift_alignment"]["rotation"],
            data["sift_alignment"]["scale"],
            data["sift_alignment"]["inliers"]
        ))
        
        # Insert detailed metrics for each pin
        for pin in data["pins"]:
            cursor.execute("""
                INSERT INTO pin_detail_metrics (inspection_id, pin_id, status, ncc, shift_px, ratio, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data["item_id"],
                pin["id"],
                pin["status"],
                pin["ncc"],
                pin["shift_px"],
                pin["ratio"],
                pin["detail"]
            ))
            
        conn.commit()
    except Exception as e:
        print(f"Database Logging Error: {e}")
        conn.rollback()
    finally:
        conn.close()

def load_config():
    """Loads configuration options from the centralized config.json."""
    config_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'config.json'))
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json ({e}). Using default settings.")
    return {}

def resolve_params(source_val):
    """Merges config.json values with custom parameters."""
    cfg = load_config()
    
    # Camera Stream parameters
    c_cfg = cfg.get("camera_stream", {})
    stability_threshold = c_cfg.get("stability_threshold", 1.2)
    reset_threshold = c_cfg.get("reset_threshold", 2.5)
    settle_duration = c_cfg.get("settle_duration_seconds", 3.0)
    cooldown_duration = c_cfg.get("cooldown_duration_seconds", 5.0)

    # Inspection Threshold parameters
    i_cfg = cfg.get("inspection_thresholds", {})
    
    inspect_config = {
        "warn_shift_thresh": i_cfg.get("warn_shift_thresh", 2.0),
        "severe_shift_thresh": i_cfg.get("severe_shift_thresh", 6.0),
        "moderate_warn_shift_thresh": i_cfg.get("moderate_warn_shift_thresh", 5.0),
        "moderate_severe_shift_thresh": i_cfg.get("moderate_severe_shift_thresh", 8.0),
        "warn_ncc_thresh": i_cfg.get("warn_ncc_thresh", 0.70),
        "severe_ncc_thresh": i_cfg.get("severe_ncc_thresh", 0.55),
        "enable_neighbor_escalation": i_cfg.get("enable_neighbor_escalation", True),
        "neighbor_shift_thresh": i_cfg.get("neighbor_shift_thresh", 2.0),
    }
    
    for key in ["missing_ncc_thresh_low", "missing_ncc_thresh_mid", "missing_ncc_thresh_high",
                "missing_ratio_thresh_low", "missing_ratio_thresh_mid", "missing_ratio_thresh_high",
                "std_thresh_high", "neighbor_ncc_thresh"]:
        if key in i_cfg:
            inspect_config[key] = i_cfg[key]

    return stability_threshold, reset_threshold, settle_duration, cooldown_duration, inspect_config, cfg

def ensure_default_profile():
    profiles_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'profiles'))
    os.makedirs(profiles_dir, exist_ok=True)
    default_profile_path = os.path.join(profiles_dir, "default_22pin.json")
    if not os.path.exists(default_profile_path):
        try:
            from src.core.inspector import PINS
            with open(default_profile_path, 'w') as f:
                json.dump(PINS, f, indent=4)
        except Exception as e:
            print(f"Error creating default profile file: {e}")

def load_profile(profile_name):
    ensure_default_profile()
    profiles_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'profiles'))
    profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading profile '{profile_name}': {e}")
    return None

def prune_snapshots(db_dir, config):
    prune_cfg = config.get("data_pruning", {})
    max_pass_images = prune_cfg.get("max_pass_images", 5)
    
    if not os.path.exists(db_dir):
        return
        
    try:
        files = [os.path.join(db_dir, f) for f in os.listdir(db_dir) if f.endswith(".png")]
        pass_files = [f for f in files if "PASS" in f]
        
        if len(pass_files) > max_pass_images:
            pass_files.sort(key=os.path.getmtime)
            to_delete = pass_files[:-max_pass_images]
            for f in to_delete:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Error deleting pruned snapshot {f}: {e}")
    except Exception as e:
        print(f"Error executing snapshot pruning policy: {e}")

def simulate_plc_handshake(status):
    """Simulates physical reject handshake over digital registers."""
    print("\n  [PLC HANDSHAKE SIMULATION]")
    if "PASS" in status:
        print("    --> Write Register 30001 (PASS)   = 1")
        print("    --> Write Register 30002 (REJECT) = 0")
        print("    --> Status Detail: PART ACCEPTED (No action required)")
    else:
        print("    --> Write Register 30001 (PASS)   = 0")
        print("    --> Write Register 30002 (REJECT) = 1")
        print("    --> Status Detail: CONVEYOR REJECT ACTUATOR TRIGGERED (Pneumatic pusher activated)")
    print()

class ThreadedCameraGrabber:
    """Thread-decoupled camera stream reader to prevent lag in live feeds."""
    def __init__(self, source):
        self.source = source
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        self.cap = cv2.VideoCapture(self.source)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = None
        
    def start(self):
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # IP Camera Reconnection or webcam timeout
                print(f"\n[Camera Grabber] Reconnecting to camera source '{self.source}'...")
                self.cap.release()
                time.sleep(2.0)
                self.cap = cv2.VideoCapture(self.source)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame.copy()
            time.sleep(0.001)
                
    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
            
    def isOpened(self):
        return self.cap.isOpened()
        
    def get(self, propId):
        return self.cap.get(propId)
        
    def release(self):
        self.stopped = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.cap.release()

class AppState:
    """Thread-safe state container shared between HMI server and vision inspection pipeline."""
    def __init__(self):
        self.lock = threading.Lock()
        self.source = "data/defect.mp4"
        self.source_changed = True  # Start True to trigger initial reader creation in background thread
        self.latest_display_frame = None
        
        # Snapshot of last inspected item to show side-by-side
        self.last_snap_frame = None
        self.last_snap_id = None
        self.last_snap_status = None
        
        self.state = "INIT"
        self.motion_score = 0.0
        self.stability_threshold = 1.2
        self.reset_threshold = 2.5
        self.settle_duration = 3.0
        
    def set_source(self, new_source):
        with self.lock:
            self.source = new_source
            self.state = "INIT"
            self.source_changed = True
            
    def get_latest_display_frame(self):
        with self.lock:
            return self.latest_display_frame.copy() if self.latest_display_frame is not None else None
            
    def set_latest_display_frame(self, frame):
        with self.lock:
            self.latest_display_frame = frame.copy() if frame is not None else None
            
    def get_last_snap_frame(self):
        with self.lock:
            return self.last_snap_frame.copy() if self.last_snap_frame is not None else None
            
    def set_last_snap(self, frame, item_id, status):
        with self.lock:
            self.last_snap_frame = frame.copy() if frame is not None else None
            self.last_snap_id = item_id
            self.last_snap_status = status

global_app_state = AppState()

class InspectionPipelineThread(threading.Thread):
    """Background thread executing the state-machine computer vision analysis loop."""
    def __init__(self, initial_source, gold_image_path):
        super().__init__()
        self.daemon = True
        self.gold_path = gold_image_path
        
        # Load Reference images
        im_gold = cv2.imread(self.gold_path)
        if im_gold is None:
            print(f"Error: Golden reference image not found at {self.gold_path}")
            sys.exit(1)
        self.gray_gold = cv2.cvtColor(im_gold, cv2.COLOR_BGR2GRAY)
        self.h_dim, self.w_dim = im_gold.shape[:2]
        self.im_gold = im_gold
        
    def run(self):
        # Local state machine parameters
        prev_gray = None
        init_timer = time.time()
        stable_start_time = None
        last_inspected_hud = None
        last_stats_str = None
        cooldown_start_time = None
        last_motion_check_time = 0.0
        
        reader = None
        is_video_file = False
        frame_delay = 0.0
        motion_history = []
        
        while True:
            # Check source change parameter updates
            with global_app_state.lock:
                source_changed = global_app_state.source_changed
                active_source = global_app_state.source
                if source_changed:
                    global_app_state.source_changed = False
                    
            if source_changed:
                if reader is not None:
                    print(f"[System] Closing previous reader for source change...")
                    reader.release()
                    reader = None
                
                # Re-resolve parameters
                stability_threshold, reset_threshold, settle_duration, cooldown_duration, inspect_config, cfg = resolve_params(active_source)
                global_app_state.stability_threshold = stability_threshold
                global_app_state.reset_threshold = reset_threshold
                global_app_state.settle_duration = settle_duration
                
                profile_name = cfg.get("active_profile", "default_22pin")
                pins = load_profile(profile_name)
                
                # Check source type
                if isinstance(active_source, str) and active_source.isdigit():
                    active_source = int(active_source)
                
                is_video_file = isinstance(active_source, str) and os.path.exists(active_source)
                if is_video_file:
                    reader = cv2.VideoCapture(active_source)
                    print(f"  [Acquisition Pipeline] Synchronous file reader initialized for '{active_source}' (Accuracy Mode)")
                    fps = reader.get(cv2.CAP_PROP_FPS)
                    if fps and fps > 0:
                        frame_delay = 1.0 / fps
                    else:
                        frame_delay = 1.0 / 30.0
                else:
                    reader = ThreadedCameraGrabber(active_source).start()
                    print(f"  [Acquisition Pipeline] Thread-decoupled grabber started for live feed '{active_source}' (Latency Reduction Mode)")
                    frame_delay = 0.001
            
            # Safety check
            if reader is None:
                time.sleep(0.05)
                continue
                
            ret, frame = reader.read()
            if not ret or frame is None:
                if is_video_file:
                    # Loop video file automatically
                    reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    time.sleep(0.01)
                    continue
                    
            # 1. Compute motion frame differencing (per-frame to ensure real-time responsiveness)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
            motion_score = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray_blurred)
                raw_motion = float(np.mean(diff))
                
                # Dynamic noise filtering: moving average over 5 frames
                motion_history.append(raw_motion)
                if len(motion_history) > 5:
                    motion_history.pop(0)
                motion_score = float(np.mean(motion_history))
            prev_gray = gray_blurred
            global_app_state.motion_score = motion_score
                
            resized_frame = cv2.resize(frame, (self.w_dim, self.h_dim))
            
            # Initialize a fallback display frame to avoid black screen dropouts
            display_frame = np.zeros((self.h_dim, self.w_dim + 340, 3), dtype=np.uint8)
            display_frame[:, :self.w_dim] = resized_frame
            display_frame[:, self.w_dim:] = 24  # dark panel
            
            # 2. State Machine transitions
            state = global_app_state.state
            
            if state == "INIT":
                cv2.putText(display_frame, "INITIALIZING SYSTEM...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                draw_status_panel(display_frame, self.w_dim, self.h_dim, "INIT", (200, 200, 200), motion_score, stability_threshold)
                
                if time.time() - init_timer >= 0.5:
                    if motion_score < stability_threshold and motion_score > 0.0:
                        H, _, stats = align_frame(self.im_gold, frame)
                        if validate_alignment(H, stats):
                            global_app_state.state = "INSPECTING"
                        else:
                            global_app_state.state = "WAITING_FOR_ITEM"
                    else:
                        global_app_state.state = "POSITIONING"
                        
            elif state == "WAITING_FOR_ITEM":
                cv2.putText(display_frame, "WAITING FOR CONNECTOR...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "Ready for next item", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                draw_status_panel(display_frame, self.w_dim, self.h_dim, "WAITING", (0, 230, 230), motion_score, stability_threshold, last_stats_str)
                
                if motion_score >= reset_threshold:
                    global_app_state.state = "POSITIONING"
                    
            elif state == "POSITIONING":
                cv2.putText(display_frame, "CONNECTOR IN MOTION...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "Stabilizing position...", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                draw_status_panel(display_frame, self.w_dim, self.h_dim, "POSITIONING", (0, 165, 255), motion_score, stability_threshold, last_stats_str)
                
                if motion_score < stability_threshold and motion_score > 0.0:
                    global_app_state.state = "STABILIZING"
                    stable_start_time = time.time()
                    
            elif state == "STABILIZING":
                elapsed = time.time() - stable_start_time
                pct = min(100.0, (elapsed / settle_duration) * 100.0)
                
                cv2.putText(display_frame, f"STABILIZING... {pct:.0f}%", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "Keep still for snap", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                draw_status_panel(display_frame, self.w_dim, self.h_dim, f"SETTLING ({pct:.0f}%)", (0, 230, 230), motion_score, stability_threshold, last_stats_str)
                
                if motion_score >= stability_threshold:
                    global_app_state.state = "POSITIONING"
                    stable_start_time = None
                elif elapsed >= settle_duration:
                    global_app_state.state = "INSPECTING"
                    
            elif state == "INSPECTING":
                print(f"\n[Inspection] Connector stabilized. Running vision inspection...")
                H, im_test_aligned, stats = align_frame(self.im_gold, frame)
                
                if validate_alignment(H, stats):
                    gray_test_aligned = cv2.cvtColor(im_test_aligned, cv2.COLOR_BGR2GRAY)
                    pin_results = inspect_frame(self.gray_gold, gray_test_aligned, config=inspect_config, pins=pins)
                    
                    total_severe_bent = sum(1 for r in pin_results if r["status"] == "SEVERE_BENT")
                    total_missing = sum(1 for r in pin_results if r["status"] == "MISSING")
                    total_warning = sum(1 for r in pin_results if r["status"] == "WARNING")
                    
                    if total_severe_bent > 0:
                        overall_status = "FAIL: BENT PINS"
                    elif total_warning > 0:
                        overall_status = "WARNING"
                    else:
                        overall_status = "PASS"
                        
                    display_frame = draw_hud(im_test_aligned, pin_results, stats)
                    
                    # Save snapshot
                    timestamp = int(time.time())
                    snapshot_status = overall_status.replace(" ", "_").replace(":", "")
                    snapshot_name = f"snap_{timestamp}_{snapshot_status}.png"
                    snapshot_path = os.path.join(DB_DIR, snapshot_name)
                    cv2.imwrite(snapshot_path, display_frame)
                    
                    # Log details
                    log_entry = {
                        "item_id": f"item_{timestamp}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                        "overall_status": overall_status,
                        "sift_alignment": {
                            "tx": float(stats["tx"]),
                            "ty": float(stats["ty"]),
                            "rotation": float(stats["rotation"]),
                            "scale": float(stats["scale"]),
                            "inliers": int(stats["inliers"])
                        },
                        "metrics": {
                            "passed_pins": sum(1 for r in pin_results if r["status"] == "PASS"),
                            "warning_pins": total_warning,
                            "severe_bent_pins": total_severe_bent,
                            "missing_pins": total_missing
                        },
                        "pins": [
                            {
                                "id": r["pin"]["id"],
                                "status": r["status"],
                                "ncc": round(r["ncc"], 3),
                                "shift_px": round(r["shift"], 2) if r["shift"] < 100 else None,
                                "ratio": round(r["ratio"], 2),
                                "detail": r["detail"]
                            } for r in pin_results
                        ],
                        "snapshot_file": snapshot_path
                    }
                    log_to_sqlite(log_entry)
                    prune_snapshots(DB_DIR, cfg)
                    simulate_plc_handshake(overall_status)
                    print_cli_report(overall_status, pin_results)
                    
                    # Update HMI Snaps display variables
                    global_app_state.set_last_snap(display_frame, f"item_{timestamp}", overall_status)
                    
                    last_stats_str = f"Status: {overall_status}\nPassed: {sum(1 for r in pin_results if r['status'] == 'PASS')}\nWarning: {total_warning}\nSevere Bent: {total_severe_bent}\nMissing: {total_missing}"
                    last_inspected_hud = display_frame.copy()
                    cooldown_start_time = time.time()
                    global_app_state.state = "RESULT_DISPLAY"
                else:
                    inliers = stats.get("inliers", 0) if stats else 0
                    scale = stats.get("scale", 1.0) if stats else 1.0
                    rot = stats.get("rotation", 0.0) if stats else 0.0
                    print(f"  [Inspection] No valid target object detected in frame (SIFT inliers: {inliers}, scale: {scale:.2f}x, rotation: {rot:.1f} deg).")
                    print("  Reverting to WAITING state.")
                    global_app_state.state = "WAITING_FOR_ITEM"
                    cv2.putText(display_frame, "ALIGNMENT FAILED", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    draw_status_panel(display_frame, self.w_dim, self.h_dim, "WAITING", (0, 230, 230), motion_score, stability_threshold, last_stats_str)
                    
            elif state == "RESULT_DISPLAY":
                if last_inspected_hud is not None:
                    display_frame = last_inspected_hud.copy()
                else:
                    cv2.putText(display_frame, "NO SNAP AVAILABLE", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    draw_status_panel(display_frame, self.w_dim, self.h_dim, "WAITING", (0, 230, 230), motion_score, stability_threshold, last_stats_str)
                    
                elapsed = time.time() - cooldown_start_time
                if elapsed >= cooldown_duration:
                    global_app_state.state = "REMOVING_ITEM"
                elif motion_score >= reset_threshold:
                    global_app_state.state = "WAITING_FOR_ITEM"
                    last_inspected_hud = None
                    prev_gray = None
                    
            elif state == "REMOVING_ITEM":
                cv2.putText(display_frame, "CONVEYOR CLEARING...", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "Conveyor running to clear zone", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
                draw_status_panel(display_frame, self.w_dim, self.h_dim, "CLEAR ZONE", (0, 165, 255), motion_score, stability_threshold, last_stats_str)
                
                if motion_score < stability_threshold:
                    if stable_start_time is None:
                        stable_start_time = time.time()
                    elif time.time() - stable_start_time >= settle_duration:
                        print("Stabilization complete, conveyor empty. Ready for next item...")
                        global_app_state.state = "WAITING_FOR_ITEM"
                else:
                    stable_start_time = None
                    
            global_app_state.set_latest_display_frame(display_frame)
            
            # Match natural frame delay / sleep configuration
            if is_video_file and frame_delay > 0:
                time.sleep(frame_delay)
            else:
                time.sleep(frame_delay)

def draw_status_panel(canvas, w_dim, h_dim, state_str, color, motion, threshold, last_stats_str=None):
    def put(txt, x, y, sc=0.4, c=(240, 240, 240), th=1):
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

    hy = 38
    put("LED PIN INSPECTOR", w_dim + 20, hy, 0.72, (255, 255, 255), 2)
    hy += 16
    put("PORTAL v6.0 (SCADA-HMI)", w_dim + 20, hy, 0.38, (120, 120, 120))
    hy += 12
    cv2.line(canvas, (w_dim + 20, hy), (w_dim + 320, hy), (60, 60, 60), 1)

    hy += 26
    put("OVERALL SYSTEM STATUS:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 12
    
    cv2.rectangle(canvas, (w_dim + 20, hy - 12), (w_dim + 320, hy + 24), (30, 30, 30), -1)
    cv2.rectangle(canvas, (w_dim + 20, hy - 12), (w_dim + 320, hy + 24), color, 1)
    put(state_str, w_dim + 30, hy + 12, 0.52, color, 2)

    hy += 50
    cv2.line(canvas, (w_dim + 20, hy), (w_dim + 320, hy), (60, 60, 60), 1)
    hy += 20
    put("REAL-TIME MOTION STATUS:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 25
    put(f"Motion Score: {motion:.3f}", w_dim + 30, hy, c=(240, 240, 240))
    hy += 20
    put(f"Settle Thresh: {threshold:.3f}", w_dim + 30, hy, c=(120, 120, 120))
    
    cv2.rectangle(canvas, (w_dim + 30, hy + 15), (w_dim + 310, hy + 25), (40, 40, 40), -1)
    bar_width = int(min(280, (motion / 4.0) * 280))
    bar_color = (0, 0, 255) if motion >= threshold else (0, 255, 0)
    cv2.rectangle(canvas, (w_dim + 30, hy + 15), (w_dim + 30 + bar_width, hy + 25), bar_color, -1)
    
    thresh_x = w_dim + 30 + int((threshold / 4.0) * 280)
    cv2.line(canvas, (thresh_x, hy + 12), (thresh_x, hy + 28), (255, 255, 255), 1)
    
    hy += 45
    cv2.line(canvas, (w_dim + 20, hy), (w_dim + 320, hy), (60, 60, 60), 1)
    hy += 20
    put("LAST INSPECTION RESULT:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 22
    if last_stats_str:
        lines = last_stats_str.split("\n")
        for line in lines:
            c = (240, 240, 240)
            if "PASS" in line:
                c = (0, 230, 0)
            elif "WARNING" in line:
                c = (0, 230, 230)
            elif "FAIL" in line:
                c = (40, 40, 255)
            put(line, w_dim + 30, hy, c=c)
            hy += 18
    else:
        put("Waiting for first inspection...", w_dim + 30, hy, c=(150, 150, 150))

def print_cli_report(overall_status, pin_results):
    total_passed = sum(1 for r in pin_results if r["status"] == "PASS")
    total_warning = sum(1 for r in pin_results if r["status"] == "WARNING")
    total_severe_bent = sum(1 for r in pin_results if r["status"] == "SEVERE_BENT")
    total_missing = sum(1 for r in pin_results if r["status"] == "MISSING")

    headers = ["Pin ID", "Status", "NCC Sim", "Shift (px)", "Int Ratio", "Inspection Detail"]
    col_widths = [10, 12, 10, 12, 12, 38]
    border = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_str = "|" + "|".join([f" {h:<{w}} " for h, w in zip(headers, col_widths)]) + "|"
    
    print("\n" + "=" * 75)
    print("                      QUALITY CONTROL CLI REPORT")
    print("=" * 75)
    print(border)
    print(header_str)
    print(border)
    for pr in pin_results:
        pid = pr["pin"]["id"]
        ncc_s = f"{pr['ncc']:.3f}"
        sh_s = f"{pr['shift']:.2f}" if pr['shift'] < 100 else "N/A"
        row = [pid, pr["status"], ncc_s, sh_s, f"{pr['ratio']:.2f}", pr["detail"]]
        row_str = "|" + "|".join([f" {str(val):<{w}} " for val, w in zip(row, col_widths)]) + "|"
        print(row_str)
    print(border)
    print(f"\nSummary:\n  Passed:       {total_passed} / 22\n  Warnings:     {total_warning} / 22\n  Severe Bent:  {total_severe_bent} / 22\n  Missing:      {total_missing} / 22")
    print("=" * 75)
    print(f"  CONNECTOR STATUS: {overall_status}")
    print("=" * 75 + "\n")

class InspectionServerHandler(http.server.BaseHTTPRequestHandler):
    """Multi-threaded Web HMI Server Handler with REST APIs and Live Video Streams."""
    def log_message(self, format, *args):
        # Suppress spammy web logs in terminal
        pass
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)
        
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        
        if path.startswith("/profiles/"):
            self.serve_profiles(path)
        elif path == "/api/stats":
            self.serve_stats(cors_headers)
        elif path == "/api/history":
            self.serve_history(cors_headers)
        elif path == "/api/pins":
            self.serve_pins(query, cors_headers)
        elif path == "/api/snapshot":
            self.serve_snapshot(query, cors_headers)
        elif path == "/api/video_feed":
            self.serve_video_feed()
        elif path == "/api/last_snap":
            self.serve_last_snap(cors_headers)
        elif path == "/api/latest_inspection":
            self.serve_latest_inspection(cors_headers)
        else:
            self.serve_static(path)
            
    def do_POST(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        
        if path == "/api/change_source":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(post_data)
                source_type = data.get("source_type", "direct")
                
                if source_type == "cp_plus":
                    ip = data.get("ip", "").strip()
                    port = data.get("port", "554").strip()
                    username = data.get("username", "").strip()
                    password = data.get("password", "").strip()
                    channel = data.get("channel", "1").strip()
                    subtype = data.get("subtype", "0").strip()
                    
                    # Construct CP Plus / Dahua standard Realmonitor RTSP URL
                    new_source = f"rtsp://{username}:{password}@{ip}:{port}/cam/realmonitor?channel={channel}&subtype={subtype}"
                elif source_type == "custom_rtsp":
                    new_source = data.get("custom_rtsp_url", "").strip()
                else:
                    new_source = data.get("source")

                if new_source:
                    global_app_state.set_source(new_source)
                    self.send_json({"status": "success", "message": f"Source changed to {new_source}"}, headers=cors_headers)
                else:
                    self.send_json({"status": "error", "message": "Missing camera source parameter(s)"}, status=400, headers=cors_headers)
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, status=500, headers=cors_headers)
                
        elif path == "/api/force_inspect":
            try:
                with global_app_state.lock:
                    global_app_state.state = "INSPECTING"
                self.send_json({"status": "success", "message": "Manual inspection triggered successfully"}, headers=cors_headers)
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, status=500, headers=cors_headers)
                
        elif path == "/api/reset_system":
            try:
                with global_app_state.lock:
                    global_app_state.state = "WAITING_FOR_ITEM"
                    global_app_state.last_snap_frame = None
                    global_app_state.last_snap_id = None
                    global_app_state.last_snap_status = None
                self.send_json({"status": "success", "message": "System status reset to WAITING"}, headers=cors_headers)
            except Exception as e:
                self.send_json({"status": "error", "message": str(e)}, status=500, headers=cors_headers)
        else:
            self.send_response(404)
            self.end_headers()

    def serve_profiles(self, path):
        profiles_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'profiles'))
        file_name = path.replace("/profiles/", "", 1)
        file_path = os.path.abspath(os.path.join(profiles_dir, file_name))
        
        if not file_path.startswith(profiles_dir):
            self.send_response(403)
            self.end_headers()
            return

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def serve_stats(self, cors_headers):
        if not os.path.exists(DB_PATH):
            self.send_json({"total": 0, "passed": 0, "warning": 0, "failed": 0}, headers=cors_headers)
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*), 
                       SUM(CASE WHEN overall_status LIKE 'PASS%' THEN 1 ELSE 0 END), 
                       SUM(CASE WHEN overall_status LIKE 'WARN%' THEN 1 ELSE 0 END), 
                       SUM(CASE WHEN overall_status LIKE 'FAIL%' THEN 1 ELSE 0 END) 
                FROM inspection_history
            """)
            total, pass_count, warn_count, fail_count = cursor.fetchone()
            
            cursor.execute("SELECT SUM(passed_pins), SUM(warning_pins), SUM(severe_bent_pins), SUM(missing_pins) FROM inspection_history")
            tot_passed, tot_warn, tot_bent, tot_missing = cursor.fetchone()
            conn.close()

            stats = {
                "total": total or 0,
                "passed": pass_count or 0,
                "warning": warn_count or 0,
                "failed": fail_count or 0,
                "pass_rate_pct": round((pass_count / total * 100.0), 1) if total and total > 0 else 0.0,
                "total_passed_pins": tot_passed or 0,
                "total_warning_pins": tot_warn or 0,
                "total_bent_pins": tot_bent or 0,
                "total_missing_pins": tot_missing or 0
            }
            self.send_json(stats, headers=cors_headers)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500, headers=cors_headers)

    def serve_history(self, cors_headers):
        if not os.path.exists(DB_PATH):
            self.send_json([], headers=cors_headers)
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, overall_status, passed_pins, warning_pins, severe_bent_pins, missing_pins, snapshot_file 
                FROM inspection_history 
                ORDER BY timestamp DESC LIMIT 50
            """)
            rows = cursor.fetchall()
            conn.close()

            history = []
            for r in rows:
                history.append({
                    "id": r[0],
                    "timestamp": r[1],
                    "overall_status": r[2],
                    "passed_pins": r[3],
                    "warning_pins": r[4],
                    "severe_bent_pins": r[5],
                    "missing_pins": r[6],
                    "snapshot_file": r[7]
                })
            self.send_json(history, headers=cors_headers)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500, headers=cors_headers)

    def serve_pins(self, query, cors_headers):
        inspection_id = query.get("id", [None])[0]
        if not inspection_id:
            self.send_json({"error": "Missing 'id' parameter"}, status=400, headers=cors_headers)
            return

        if not os.path.exists(DB_PATH):
            self.send_json([], headers=cors_headers)
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pin_id, status, ncc, shift_px, ratio, detail 
                FROM pin_detail_metrics 
                WHERE inspection_id = ?
            """, (inspection_id,))
            rows = cursor.fetchall()
            conn.close()

            pins = []
            for r in rows:
                pins.append({
                    "id": r[0],
                    "status": r[1],
                    "ncc": r[2],
                    "shift_px": r[3],
                    "ratio": r[4],
                    "detail": r[5]
                })
            self.send_json(pins, headers=cors_headers)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500, headers=cors_headers)

    def serve_snapshot(self, query, cors_headers):
        inspection_id = query.get("id", [None])[0]
        if not inspection_id:
            self.send_json({"error": "Missing 'id' parameter"}, status=400, headers=cors_headers)
            return

        if not os.path.exists(DB_PATH):
            self.send_response(404)
            self.end_headers()
            return

        try:
            conn = connect_db(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT snapshot_file FROM inspection_history WHERE id = ?", (inspection_id,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                file_path = os.path.abspath(row[0])
                if file_path.startswith(DB_DIR) and os.path.exists(file_path):
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        self.wfile.write(f.read())
                    return
            self.send_response(404)
            self.end_headers()
        except Exception as e:
            self.send_response(500)
            self.end_headers()

    def serve_video_feed(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                frame = global_app_state.get_latest_display_frame()
                if frame is None:
                    # Draw a nice SCADA loading block
                    frame = np.zeros((480, 640 + 340, 3), dtype=np.uint8)
                    cv2.putText(frame, "CONNECTING SOURCE...", (230, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                time.sleep(0.04)  # Cap stream to ~25fps
        except Exception:
            pass

    def serve_last_snap(self, cors_headers):
        frame = global_app_state.get_last_snap_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                return
                
        # Send placeholder if no snaps have been generated yet
        placeholder_path = os.path.join(DATA_DIR, "Good Pins.png")
        if os.path.exists(placeholder_path):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            with open(placeholder_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def serve_latest_inspection(self, cors_headers):
        self.send_json({
            "id": global_app_state.last_snap_id,
            "status": global_app_state.last_snap_status
        }, headers=cors_headers)

    def serve_static(self, path):
        if path == "/":
            path = "/index.html"
            
        file_path = os.path.abspath(os.path.join(DASHBOARD_DIR, path.lstrip("/")))
        if not file_path.startswith(DASHBOARD_DIR):
            self.send_response(403)
            self.end_headers()
            return

        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            content_type = "text/html"
            if file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".css"):
                content_type = "text/css"
            elif file_path.endswith(".json"):
                content_type = "application/json"
            elif file_path.endswith(".png"):
                content_type = "image/png"
            elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
                content_type = "image/jpeg"

            self.send_header("Content-Type", content_type)
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data, status=200, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server that handles concurrent dashboard client requests."""
    daemon_threads = True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-time LED Pin Quality Control Web HMI Portal")
    parser.add_argument("--port", type=int, default=8000, help="Web server HMI port. Default is 8000.")
    parser.add_argument("--source", type=str, default="data/defect.mp4", help="Video source (index, RTSP link, or video file path).")
    parser.add_argument("--gold", type=str, default="data/Good Pins.png", help="Path to golden reference image.")
    args = parser.parse_args()

    # Initialize Database
    init_db()

    # Start vision processor background thread
    print(f"\n[System] Initializing background Vision Inspection Pipeline...")
    pipeline_thread = InspectionPipelineThread(args.source, args.gold)
    pipeline_thread.start()
    
    # Start HMI dashboard web server
    port = args.port
    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, InspectionServerHandler)
    
    print(f"\n============================================================")
    print(f"      LED PIN INSPECTION UNIFIED WEB HMI PORTAL v6.0")
    print(f"============================================================")
    print(f"  Live Server Running at: http://localhost:{port}")
    print(f"  SQLite Logging DB:      {DB_PATH}")
    print(f"  Conveyor Grabber:       {args.source}")
    print(f"  Press Ctrl+C to shutdown.")
    print(f"============================================================\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[System] Shutdown signal received. Exiting gracefully...")
        sys.exit(0)

if __name__ == "__main__":
    main()
