import os
import sys

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add workspace root (two levels up from src/core) to python path to allow src.core.* imports
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..')))

import cv2
import numpy as np
import argparse
import time
import json
import sqlite3
from src.core.inspector import align_frame, warp_with_homography, inspect_frame, draw_hud

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'data'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'results'))
DB_DIR = os.path.join(RESULTS_DIR, 'db_snapshots')
SQLITE_DB_PATH = os.path.join(RESULTS_DIR, 'inspection.db')

# Ensure database snapshot directory exists
os.makedirs(DB_DIR, exist_ok=True)

def init_db():
    """Initializes SQLite database schema for enterprise logging."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    # 1. Create inspection history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspection_history (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            overall_status TEXT,
            tx REAL,
            ty REAL,
            rotation REAL,
            scale REAL,
            inliers INTEGER,
            passed_pins INTEGER,
            warning_pins INTEGER,
            severe_bent_pins INTEGER,
            missing_pins INTEGER,
            snapshot_file TEXT
        )
    """)
    
    # 2. Create pin results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pin_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id TEXT,
            pin_id TEXT,
            status TEXT,
            ncc REAL,
            shift_px REAL,
            ratio REAL,
            detail TEXT,
            FOREIGN KEY (inspection_id) REFERENCES inspection_history (id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

def log_to_sqlite(data):
    """Logs inspection and detailed pin metrics to local SQLite database."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Insert into inspection_history
        cursor.execute("""
            INSERT OR REPLACE INTO inspection_history (
                id, timestamp, overall_status, tx, ty, rotation, scale, inliers,
                passed_pins, warning_pins, severe_bent_pins, missing_pins, snapshot_file
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["item_id"],
            data["timestamp"],
            data["overall_status"],
            data["sift_alignment"]["tx"],
            data["sift_alignment"]["ty"],
            data["sift_alignment"]["rotation"],
            data["sift_alignment"]["scale"],
            data["sift_alignment"]["inliers"],
            data["metrics"]["passed_pins"],
            data["metrics"]["warning_pins"],
            data["metrics"]["severe_bent_pins"],
            data["metrics"]["missing_pins"],
            data["snapshot_file"]
        ))
        
        # Insert pins
        for pin in data["pins"]:
            cursor.execute("""
                INSERT INTO pin_results (
                    inspection_id, pin_id, status, ncc, shift_px, ratio, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
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
        print(f"  Logged database entry to SQLite: {data['item_id']}")
    except Exception as e:
        print(f"Error logging to SQLite DB: {e}")
        conn.rollback()
    finally:
        conn.close()

def load_config():
    """Loads configuration parameters from external config.json file in root workspace."""
    config_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'config.json'))
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                print(f"Loaded configuration parameters from {config_path}")
                return cfg
        except Exception as e:
            print(f"Warning: Failed to load config.json ({e}). Using default settings.")
    else:
        print("config.json not found in workspace root. Using default CLI arguments.")
    return {}

def resolve_params(args):
    """Merges config.json values with CLI overrides."""
    cfg = load_config()
    
    # Camera Stream parameters
    c_cfg = cfg.get("camera_stream", {})
    stability_threshold = c_cfg.get("stability_threshold", args.thresh) if args.thresh == 1.2 else args.thresh
    reset_threshold = c_cfg.get("reset_threshold", args.reset_thresh) if args.reset_thresh == 2.5 else args.reset_thresh
    settle_duration = c_cfg.get("settle_duration_seconds", args.settle) if args.settle == 3.0 else args.settle
    cooldown_duration = c_cfg.get("cooldown_duration_seconds", args.cooldown) if args.cooldown == 5.0 else args.cooldown

    # Inspection Threshold parameters
    i_cfg = cfg.get("inspection_thresholds", {})
    
    inspect_config = {
        "warn_shift_thresh": i_cfg.get("warn_shift_thresh", args.warn_shift) if args.warn_shift == 2.0 else args.warn_shift,
        "severe_shift_thresh": i_cfg.get("severe_shift_thresh", args.severe_shift) if args.severe_shift == 6.0 else args.severe_shift,
        "moderate_warn_shift_thresh": i_cfg.get("moderate_warn_shift_thresh", args.mod_warn_shift) if args.mod_warn_shift == 5.0 else args.mod_warn_shift,
        "moderate_severe_shift_thresh": i_cfg.get("moderate_severe_shift_thresh", args.mod_severe_shift) if args.mod_severe_shift == 8.0 else args.mod_severe_shift,
        "warn_ncc_thresh": i_cfg.get("warn_ncc_thresh", args.warn_ncc) if args.warn_ncc == 0.70 else args.warn_ncc,
        "severe_ncc_thresh": i_cfg.get("severe_ncc_thresh", args.severe_ncc) if args.severe_ncc == 0.55 else args.severe_ncc,
        "enable_neighbor_escalation": i_cfg.get("enable_neighbor_escalation", not args.disable_neighbor) if not args.disable_neighbor else False,
        "neighbor_shift_thresh": i_cfg.get("neighbor_shift_thresh", args.neighbor_shift) if args.neighbor_shift == 2.0 else args.neighbor_shift,
    }
    
    # Copy other metrics from i_cfg if present, else use defaults in inspector
    for key in ["missing_ncc_thresh_low", "missing_ncc_thresh_mid", "missing_ncc_thresh_high",
                "missing_ratio_thresh_low", "missing_ratio_thresh_mid", "missing_ratio_thresh_high",
                "std_thresh_high", "neighbor_ncc_thresh"]:
        if key in i_cfg:
            inspect_config[key] = i_cfg[key]

    return stability_threshold, reset_threshold, settle_duration, cooldown_duration, inspect_config, cfg

def ensure_default_profile():
    """Ensures that the default profiles directory and default profile JSON exist."""
    profiles_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'profiles'))
    os.makedirs(profiles_dir, exist_ok=True)
    
    default_profile_path = os.path.join(profiles_dir, "default_22pin.json")
    if not os.path.exists(default_profile_path):
        try:
            from src.core.inspector import PINS
            with open(default_profile_path, 'w') as f:
                json.dump(PINS, f, indent=4)
            print(f"Created default product profile file at {default_profile_path}")
        except Exception as e:
            print(f"Error creating default profile file: {e}")

def load_profile(profile_name):
    """Loads the pin coordinates configuration for a given product profile."""
    ensure_default_profile()
    profiles_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'profiles'))
    profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
    
    if os.path.exists(profile_path):
        try:
            with open(profile_path, 'r') as f:
                pins = json.load(f)
                print(f"Loaded product profile '{profile_name}' with {len(pins)} pins.")
                return pins
        except Exception as e:
            print(f"Error loading profile '{profile_name}': {e}. Using default template.")
    else:
        print(f"Profile file '{profile_path}' not found. Using default template.")
    return None

def prune_snapshots(db_dir, config):
    """Prunes stored snapshots according to the configured disk-space policy."""
    prune_cfg = config.get("data_pruning", {})
    max_pass_images = prune_cfg.get("max_pass_images", 5)
    max_age_hours = prune_cfg.get("max_pass_age_hours", 24)
    
    if not os.path.exists(db_dir):
        return
        
    pass_files = []
    now = time.time()
    
    for filename in os.listdir(db_dir):
        if not filename.startswith("snap_") or not filename.endswith(".png"):
            continue
        file_path = os.path.join(db_dir, filename)
        
        # Keep all FAIL images for forensics
        if "_FAIL_" in filename:
            continue
            
        # Check file age
        file_age_hours = (now - os.path.getmtime(file_path)) / 3600.0
        if file_age_hours > max_age_hours:
            try:
                os.remove(file_path)
                print(f"  [Pruned] Deleted old snapshot due to age (> {max_age_hours}h): {filename}")
            except Exception as e:
                print(f"  [Pruned] Error deleting {filename}: {e}")
            continue
            
        pass_files.append((file_path, os.path.getmtime(file_path)))
        
    # Sort files by modification time (oldest first)
    pass_files.sort(key=lambda x: x[1])
    
    # If the count exceeds max_pass_images, prune the oldest ones
    if len(pass_files) > max_pass_images:
        to_delete_count = len(pass_files) - max_pass_images
        for i in range(to_delete_count):
            file_path = pass_files[i][0]
            try:
                os.remove(file_path)
                print(f"  [Pruned] Deleted old snapshot to enforce limit (max {max_pass_images}): {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  [Pruned] Error deleting {file_path}: {e}")

import threading

class ThreadedCameraGrabber:
    """Thread-safe camera grabber that continuously fetches frames in a background thread
    to prevent buffer build-up on live streams (webcams and RTSP feeds).
    """
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
        
    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            with self.lock:
                self.ret = ret
                self.frame = frame
                
    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)
            
    def isOpened(self):
        return self.cap.isOpened()
        
    def get(self, prop_id):
        return self.cap.get(prop_id)
        
    def release(self):
        self.stopped = True
        self.cap.release()

def simulate_plc_handshake(overall_status):
    """Simulates register writes to a connected PLC for reject hardware integration.
    - PASS status: write 1 to Register 30001 (PASS_TRIGGER), 0 to 30002 (REJECT_TRIGGER)
    - FAIL status: write 0 to Register 30001, 1 to 30002
    - WARNING status: write 1 to 30001, 0 to 30002
    """
    pass_reg = 30001
    reject_reg = 30002
    
    if "FAIL" in overall_status:
        pass_val = 0
        reject_val = 1
        action_detail = "CONVEYOR REJECT ACTUATOR TRIGGERED (Pneumatic pusher activated)"
    else:
        pass_val = 1
        reject_val = 0
        action_detail = "CONVEYOR PASS REGISTER ACTIVE (Part proceeds on conveyor)"
        
    print(f"\n  [PLC HANDSHAKE SIMULATION]")
    print(f"    --> Write Register {pass_reg} (PASS)   = {pass_val}")
    print(f"    --> Write Register {reject_reg} (REJECT) = {reject_val}")
    print(f"    --> Status Detail: {action_detail}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Real-Time LED Pin Inspection v2.")
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="0",
        help="Camera index, video path, or RTSP stream URL."
    )
    parser.add_argument(
        "--gold", "-g",
        type=str,
        default=os.path.join(DATA_DIR, "New_Gold.jpg"),
        help="Path to golden reference image."
    )
    parser.add_argument(
        "--thresh", "-t",
        type=float,
        default=1.2,
        help="Stability motion difference threshold. Lower = more sensitive."
    )
    parser.add_argument(
        "--reset-thresh", "-r",
        type=float,
        default=2.5,
        help="Motion difference threshold to reset display and detect a new item."
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="Settle duration (in seconds) of zero-motion before snapping."
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Cooldown lockout duration (in seconds) after capturing a snapshot."
    )
    # Inspection logic parameters
    parser.add_argument(
        "--warn-shift",
        type=float,
        default=2.0,
        help="Shift threshold (in pixels) for minor warning classifications. Default is 2.0."
    )
    parser.add_argument(
        "--severe-shift",
        type=float,
        default=6.0,
        help="Shift threshold (in pixels) for severe bent classifications. Default is 6.0."
    )
    parser.add_argument(
        "--mod-warn-shift",
        type=float,
        default=5.0,
        help="Shift threshold for moderate NCC warning classifications. Default is 5.0."
    )
    parser.add_argument(
        "--mod-severe-shift",
        type=float,
        default=8.0,
        help="Shift threshold for moderate NCC severe bent classifications. Default is 8.0."
    )
    parser.add_argument(
        "--warn-ncc",
        type=float,
        default=0.70,
        help="NCC threshold below which warnings can occur. Default is 0.70."
    )
    parser.add_argument(
        "--severe-ncc",
        type=float,
        default=0.55,
        help="NCC threshold below which severe bent is flagged. Default is 0.55."
    )
    parser.add_argument(
        "--disable-neighbor",
        action="store_true",
        help="Disable adjacent neighbor failure escalation."
    )
    parser.add_argument(
        "--neighbor-shift",
        type=float,
        default=2.0,
        help="Shift threshold for neighbor escalation warnings. Default is 2.0."
    )
    return parser.parse_args()

def draw_status_panel(canvas, w_dim, h_dim, state_str, color, motion, threshold, last_stats_str=None):
    """Renders the status dashboard content during positioning, stabilizing, and waiting states."""
    def put(txt, x, y, sc=0.4, c=(240, 240, 240), th=1):
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

    put("LED PIN INSPECTOR", w_dim + 20, 38, 0.72, (255, 255, 255), 2)
    put("PORTAL v6.0 (ROBUST-FLOW)", w_dim + 20, 54, 0.38, (120, 120, 120))
    cv2.line(canvas, (w_dim + 20, 66), (w_dim + 320, 66), (60, 60, 60), 1)

    hy = 92
    put("AUTO-INSPECTION STATE:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 24
    
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
    
    # Draw motion indicator bar
    cv2.rectangle(canvas, (w_dim + 30, hy + 15), (w_dim + 310, hy + 25), (40, 40, 40), -1)
    bar_width = int(min(280, (motion / 4.0) * 280))
    bar_color = (0, 0, 255) if motion >= threshold else (0, 255, 0)
    cv2.rectangle(canvas, (w_dim + 30, hy + 15), (w_dim + 30 + bar_width, hy + 25), bar_color, -1)
    
    # Threshold indicator vertical line
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
    """Outputs a clean inspection breakdown to the CLI terminal."""
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
    
    print(f"\nSummary:")
    print(f"  Passed:      {total_passed:>2d} / 22")
    print(f"  Warnings:    {total_warning:>2d} / 22")
    print(f"  Severe Bent: {total_severe_bent:>2d} / 22")
    print(f"  Missing:     {total_missing:>2d} / 22")
    print("=" * 75)
    print(f"  CONNECTOR STATUS: {overall_status}")
    print("=" * 75 + "\n")

def main():
    args = parse_args()
    init_db()

    im_gold = cv2.imread(args.gold)
    if im_gold is None:
        print(f"Error: Golden reference image not found at {args.gold}")
        sys.exit(1)

    gray_gold = cv2.cvtColor(im_gold, cv2.COLOR_BGR2GRAY)
    h_dim, w_dim = im_gold.shape[:2]

    source = args.source
    if source.isdigit():
        source = int(source)

    is_video_file = isinstance(source, str) and os.path.exists(source)
    if is_video_file:
        reader = cv2.VideoCapture(source)
        print(f"  [Acquisition Pipeline] Synchronous file reader initialized for '{source}' (Accuracy Mode)")
    else:
        reader = ThreadedCameraGrabber(source).start()
        print(f"  [Acquisition Pipeline] Thread-decoupled grabber started for live feed '{source}' (Latency Reduction Mode)")

    if not reader.isOpened():
        print(f"Error: Could not open source: {source}")
        sys.exit(1)

    frame_delay = 0.0
    if is_video_file:
        fps = reader.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            frame_delay = 1.0 / fps
        else:
            frame_delay = 1.0 / 30.0

    state = "INIT"
    stable_start_time = None
    last_inspected_hud = None
    last_stats_str = None
    cooldown_start_time = None
    last_sift_check_time = 0.0
    
    prev_gray = None
    init_timer = time.time()
    
    stability_threshold, reset_threshold, settle_duration, cooldown_duration, inspect_config, cfg = resolve_params(args)

    profile_name = cfg.get("active_profile", "default_22pin")
    pins = load_profile(profile_name)

    print("\n" + "=" * 60)
    print("     AUTOMATED REAL-TIME LED PIN INSPECTOR HUD v2")
    print("=" * 60)
    print(f"  Active Profile:       {profile_name}")
    print(f"  Stability Threshold:   {stability_threshold}")
    print(f"  Reset Motion Thresh:  {reset_threshold}")
    print(f"  Settle Time Needed:   {settle_duration} sec")
    print(f"  Warn Shift Threshold: {inspect_config['warn_shift_thresh']} px")
    print(f"  Severe Shift Thresh:  {inspect_config['severe_shift_thresh']} px")
    print(f"  Neighbor Escalation:  {'Enabled' if inspect_config['enable_neighbor_escalation'] else 'Disabled'}")
    print("  Controls:")
    print("    [R] - Force recalculation / manual reset")
    print("    [Q] - Quit")
    print("=" * 60 + "\n")

    window_name = "Automated LED Pin Inspection HUD v2"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = reader.read()
        if not ret:
            print("Stream ended or frame could not be read.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_score = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray_blurred)
            motion_score = float(np.mean(diff))
        prev_gray = gray_blurred

        resized_frame = cv2.resize(frame, (w_dim, h_dim))
        display_frame = None

        if state == "INIT":
            display_frame = np.zeros((h_dim, w_dim + 340, 3), dtype=np.uint8)
            display_frame[:, :w_dim] = resized_frame
            display_frame[:, w_dim:] = 24
            cv2.putText(display_frame, "INITIALIZING SYSTEM...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            draw_status_panel(display_frame, w_dim, h_dim, "INIT", (200, 200, 200), motion_score, stability_threshold)
            
            if time.time() - init_timer >= 0.5:
                if motion_score < stability_threshold and motion_score > 0.0:
                    H, _, _ = align_frame(im_gold, frame)
                    if H is not None:
                        state = "INSPECTING"
                    else:
                        state = "WAITING_FOR_ITEM"
                else:
                    state = "POSITIONING"

        elif state == "WAITING_FOR_ITEM":
            display_frame = np.zeros((h_dim, w_dim + 340, 3), dtype=np.uint8)
            display_frame[:, :w_dim] = resized_frame
            display_frame[:, w_dim:] = 24
            
            cv2.putText(display_frame, "WAITING FOR CONNECTOR...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 230), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Ready for next item", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            
            draw_status_panel(display_frame, w_dim, h_dim, "WAITING", (0, 230, 230), motion_score, stability_threshold, last_stats_str)
            
            if motion_score >= reset_threshold:
                state = "POSITIONING"

        elif state == "POSITIONING":
            display_frame = np.zeros((h_dim, w_dim + 340, 3), dtype=np.uint8)
            display_frame[:, :w_dim] = resized_frame
            display_frame[:, w_dim:] = 24
            
            cv2.putText(display_frame, "POSITIONING CONNECTOR...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Aligning housing in view", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            
            draw_status_panel(display_frame, w_dim, h_dim, "POSITIONING", (230, 230, 0), motion_score, stability_threshold, last_stats_str)
            
            if motion_score < stability_threshold and motion_score > 0.0:
                state = "STABILIZING"
                stable_start_time = time.time()

        elif state == "STABILIZING":
            display_frame = np.zeros((h_dim, w_dim + 340, 3), dtype=np.uint8)
            display_frame[:, :w_dim] = resized_frame
            display_frame[:, w_dim:] = 24
            
            elapsed = time.time() - stable_start_time
            pct = min(100.0, (elapsed / settle_duration) * 100.0)
            
            cv2.putText(display_frame, f"STABILIZING... {pct:.0f}%", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 230), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Keep still for snap", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            
            draw_status_panel(display_frame, w_dim, h_dim, f"SETTLING ({pct:.0f}%)", (0, 230, 230), motion_score, stability_threshold, last_stats_str)

            if motion_score >= stability_threshold:
                state = "POSITIONING"
                stable_start_time = None
            elif elapsed >= settle_duration:
                state = "INSPECTING"

        elif state == "INSPECTING":
            print(f"\nConnector stabilized! Capturing snap and running QC pipeline...")
            H, im_test_aligned, stats = align_frame(im_gold, frame)
            
            if H is not None:
                gray_test_aligned = cv2.cvtColor(im_test_aligned, cv2.COLOR_BGR2GRAY)
                pin_results = inspect_frame(gray_gold, gray_test_aligned, config=inspect_config, pins=pins)
                
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
                
                timestamp = int(time.time())
                snapshot_status = overall_status.replace(" ", "_").replace(":", "")
                snapshot_name = f"snap_{timestamp}_{snapshot_status}.png"
                snapshot_path = os.path.join(DB_DIR, snapshot_name)
                cv2.imwrite(snapshot_path, display_frame)
                print(f"  Saved snapshot: {snapshot_path}")
                
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
                
                last_stats_str = f"Status: {overall_status}\nPassed: {sum(1 for r in pin_results if r['status'] == 'PASS')}\nWarning: {total_warning}\nSevere Bent: {total_severe_bent}\nMissing: {total_missing}"
                last_inspected_hud = display_frame.copy()
                cooldown_start_time = time.time()
                state = "RESULT_DISPLAY"
            else:
                print("  Warning: Alignment failed on static snap. False trigger. Returning to WAITING.")
                state = "WAITING_FOR_ITEM"

        elif state == "RESULT_DISPLAY":
            display_frame = last_inspected_hud.copy()
            elapsed_cooldown = time.time() - cooldown_start_time
            remaining = max(0.0, cooldown_duration - elapsed_cooldown)
            
            if remaining > 0:
                cv2.putText(display_frame, f"COOLDOWN: {remaining:.1f}s remaining", (20, h_dim - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(display_frame, "Lockout active - wait to remove", (20, h_dim - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 250), 1, cv2.LINE_AA)
                
                cv2.rectangle(display_frame, (w_dim + 20, h_dim - 35), (w_dim + 320, h_dim - 10), (30, 30, 30), -1)
                cv2.rectangle(display_frame, (w_dim + 20, h_dim - 35), (w_dim + 320, h_dim - 10), (0, 165, 255), 1)
                cv2.putText(display_frame, "STATE: COOLDOWN LOCKOUT", (w_dim + 30, h_dim - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(display_frame, f"LIVE MOTION: {motion_score:.2f} (TH: {reset_threshold})", (20, h_dim - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display_frame, "Remove item to inspect next", (20, h_dim - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 120), 1, cv2.LINE_AA)

                cv2.rectangle(display_frame, (w_dim + 20, h_dim - 35), (w_dim + 320, h_dim - 10), (30, 30, 30), -1)
                cv2.rectangle(display_frame, (w_dim + 20, h_dim - 35), (w_dim + 320, h_dim - 10), (0, 255, 0), 1)
                cv2.putText(display_frame, "STATE: READY TO REMOVE", (w_dim + 30, h_dim - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                if motion_score >= reset_threshold:
                    print("Large motion detected (connector removed). Transitioning to REMOVING...")
                    state = "REMOVING_ITEM"

        elif state == "REMOVING_ITEM":
            display_frame = np.zeros((h_dim, w_dim + 340, 3), dtype=np.uint8)
            display_frame[:, :w_dim] = resized_frame
            display_frame[:, w_dim:] = 24
            
            cv2.putText(display_frame, "REMOVING CONNECTOR...", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Waiting for stabilization", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
            
            draw_status_panel(display_frame, w_dim, h_dim, "REMOVING", (0, 0, 255), motion_score, stability_threshold, last_stats_str)
            
            if motion_score < stability_threshold and motion_score > 0.0:
                now = time.time()
                if now - last_sift_check_time >= 1.0:
                    last_sift_check_time = now
                    H, _, _ = align_frame(im_gold, frame)
                    if H is None:
                        print("Stabilization complete, conveyor empty. Ready for next item...")
                        state = "WAITING_FOR_ITEM"

        cv2.imshow(window_name, display_frame)

        if is_video_file and frame_delay > 0:
            time.sleep(frame_delay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('r') or key == ord('R'):
            print("Manual reset triggered...")
            state = "WAITING_FOR_ITEM"
            last_inspected_hud = None
            prev_gray = None

    reader.release()
    cv2.destroyAllWindows()
    print("Automated stream finished.")

if __name__ == '__main__':
    main()
