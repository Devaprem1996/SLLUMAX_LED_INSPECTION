import cv2
import numpy as np
import math

PINS = [
    {"id": "C0_R0", "cx": 208, "cy": 254, "w": 28, "h": 28},
    {"id": "C0_R1", "cx": 207, "cy": 310, "w": 28, "h": 28},
    {"id": "C0_R2", "cx": 207, "cy": 360, "w": 28, "h": 28},
    {"id": "C0_R3", "cx": 208, "cy": 415, "w": 28, "h": 28},
    {"id": "C1_R0", "cx": 257, "cy": 207, "w": 24, "h": 24},
    {"id": "C1_R1", "cx": 259, "cy": 238, "w": 24, "h": 24},
    {"id": "C1_R2", "cx": 262, "cy": 279, "w": 24, "h": 24},
    {"id": "C1_R3", "cx": 261, "cy": 317, "w": 24, "h": 24},
    {"id": "C1_R4", "cx": 261, "cy": 354, "w": 24, "h": 24},
    {"id": "C1_R5", "cx": 260, "cy": 391, "w": 24, "h": 24},
    {"id": "C1_R6", "cx": 257, "cy": 432, "w": 24, "h": 24},
    {"id": "C2_R0", "cx": 333, "cy": 207, "w": 24, "h": 24},
    {"id": "C2_R1", "cx": 334, "cy": 238, "w": 24, "h": 24},
    {"id": "C2_R2", "cx": 337, "cy": 280, "w": 24, "h": 24},
    {"id": "C2_R3", "cx": 337, "cy": 317, "w": 24, "h": 24},
    {"id": "C2_R4", "cx": 336, "cy": 354, "w": 24, "h": 24},
    {"id": "C2_R5", "cx": 336, "cy": 392, "w": 24, "h": 24},
    {"id": "C2_R6", "cx": 333, "cy": 433, "w": 24, "h": 24},
    {"id": "C3_R0", "cx": 395, "cy": 256, "w": 28, "h": 28},
    {"id": "C3_R1", "cx": 393, "cy": 311, "w": 28, "h": 28},
    {"id": "C3_R2", "cx": 392, "cy": 362, "w": 28, "h": 28},
    {"id": "C3_R3", "cx": 394, "cy": 419, "w": 28, "h": 28},
]

def get_subpixel_tip(roi, noise_floor=110):
    """Isolates the shiny pin tip peak via dynamic 92% intensity segmentation."""
    max_val = np.max(roi)
    if max_val < noise_floor:
        return None
    thresh_val = max(noise_floor, int(0.92 * max_val))
    _, t_roi = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
    m = cv2.moments(t_roi)
    if m['m00'] > 0:
        return (m['m10'] / m['m00'], m['m01'] / m['m00'])
    else:
        _, _, _, max_loc = cv2.minMaxLoc(roi)
        return (float(max_loc[0]), float(max_loc[1]))

_cached_gold_gray = None
_cached_kp_gold = None
_cached_des_gold = None

def align_frame(im_gold, im_test):
    """Performs SIFT keypoint detection and alignment between reference and test target."""
    global _cached_gold_gray, _cached_kp_gold, _cached_des_gold

    gray_test = cv2.cvtColor(im_test, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=1500)

    gold_gray = cv2.cvtColor(im_gold, cv2.COLOR_BGR2GRAY)
    if _cached_gold_gray is None or not np.array_equal(_cached_gold_gray, gold_gray):
        _cached_gold_gray = gold_gray
        _cached_kp_gold, _cached_des_gold = sift.detectAndCompute(_cached_gold_gray, None)

    kp_gold, des_gold = _cached_kp_gold, _cached_des_gold
    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    if des_gold is None or des_test is None:
        return None, None, {}

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )
    knn_matches = flann.knnMatch(des_gold, des_test, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 10:
        return None, None, {}

    pts_gold = np.float32([kp_gold[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_test = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_test, pts_gold, cv2.RANSAC, 5.0)
    if H is None:
        return None, None, {}

    h_dim, w_dim = im_gold.shape[:2]
    im_test_aligned = cv2.warpPerspective(im_test, H, (w_dim, h_dim))

    # Decompose H
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    global_tx, global_ty = H[0, 2], H[1, 2]
    global_scale = (math.sqrt(a**2 + c**2) + math.sqrt(b**2 + d**2)) / 2.0
    global_rotation = math.atan2(c, a) * 180.0 / math.pi

    stats = {
        "tx": global_tx,
        "ty": global_ty,
        "rotation": global_rotation,
        "scale": global_scale,
        "inliers": int(np.sum(mask)),
        "total_matches": len(good_matches)
    }
    return H, im_test_aligned, stats

def warp_with_homography(im_gold, im_test, H):
    """Warps testing frame directly using precomputed homography matrix H (cached alignment)."""
    h_dim, w_dim = im_gold.shape[:2]
    im_test_aligned = cv2.warpPerspective(im_test, H, (w_dim, h_dim))

    # Decompose H
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    global_tx, global_ty = H[0, 2], H[1, 2]
    global_scale = (math.sqrt(a**2 + c**2) + math.sqrt(b**2 + d**2)) / 2.0
    global_rotation = math.atan2(c, a) * 180.0 / math.pi

    stats = {
        "tx": global_tx,
        "ty": global_ty,
        "rotation": global_rotation,
        "scale": global_scale,
        "inliers": 0,
        "total_matches": 0
    }
    return im_test_aligned, stats

DEFAULT_CONFIG = {
    # Shift thresholds
    "warn_shift_thresh": 2.0,             # Increased from 1.5 to reduce sensitivity
    "severe_shift_thresh": 6.0,
    "moderate_warn_shift_thresh": 5.0,    # Increased from 4.5 to reduce sensitivity
    "moderate_severe_shift_thresh": 8.0,
    
    # NCC thresholds
    "warn_ncc_thresh": 0.70,
    "severe_ncc_thresh": 0.55,
    "missing_ncc_thresh_low": 0.58,
    "missing_ncc_thresh_mid": 0.60,
    "missing_ncc_thresh_high": 0.70,
    
    # Intensity ratio thresholds
    "missing_ratio_thresh_low": 0.40,
    "missing_ratio_thresh_mid": 0.55,
    "missing_ratio_thresh_high": 0.75,
    "std_thresh_high": 22.0,
    
    # Neighbor context escalation
    "enable_neighbor_escalation": True,
    "neighbor_shift_thresh": 2.0,         # Increased from 1.5 to reduce sensitivity
    "neighbor_ncc_thresh": 0.70,
}

def inspect_frame(gray_gold, gray_test_aligned, config=None, pins=None):
    """Executes the multi-pass quality control checks on the aligned frame."""
    cfg = DEFAULT_CONFIG.copy()
    if config is not None:
        cfg.update(config)

    pin_results = []
    active_pins = pins if pins is not None else PINS

    for pin in active_pins:
        pid = pin["id"]
        cx, cy, w, h = pin["cx"], pin["cy"], pin["w"], pin["h"]
        x1, y1 = cx - w // 2, cy - h // 2

        # Grayscale crops
        g_roi = gray_gold[y1:y1 + h, x1:x1 + w]
        t_roi = gray_test_aligned[y1:y1 + h, x1:x1 + w]

        # Intensity ratio (presence check)
        g_nh = gray_gold[cy - 2:cy + 3, cx - 2:cx + 3]
        t_nh = gray_test_aligned[cy - 2:cy + 3, cx - 2:cx + 3]
        g_mean = np.mean(g_nh)
        t_mean = np.mean(t_nh)
        ratio = t_mean / g_mean if g_mean > 0 else 0.0
        t_std = float(np.std(t_nh))

        # NCC pattern similarity
        if np.std(g_roi) == 0 or np.std(t_roi) == 0:
            ncc = 0.0
        else:
            ncc = float(cv2.matchTemplate(t_roi, g_roi, cv2.TM_CCOEFF_NORMED)[0][0])

        # Sub-pixel tip shift
        g_tip = get_subpixel_tip(g_roi)
        t_tip = get_subpixel_tip(t_roi)
        shift = 0.0
        if g_tip and t_tip:
            shift = math.sqrt((g_tip[0] - t_tip[0]) ** 2 + (g_tip[1] - t_tip[1]) ** 2)

        # Classification separating MISSING and SEVERE_BENT
        status = "PASS"
        detail = "Pin Ok"

        # A pin is only MISSING if the intensity is extremely low AND the pattern correlation is high.
        # If the pattern is heavily degraded (NCC < severe_ncc_thresh), it indicates physical structural deformation (a bent pin)
        # rather than a completely empty slot.
        if (ratio < cfg["missing_ratio_thresh_low"] and ncc >= cfg["missing_ncc_thresh_low"]) or g_tip is None:
            status = "MISSING"
            detail = f"MISSING: Missing Pin (Ratio {ratio:.2f})"
            shift = 99.9
        elif ratio < cfg["missing_ratio_thresh_mid"] and ncc >= cfg["missing_ncc_thresh_mid"]:
            status = "MISSING"
            detail = f"MISSING: Missing Pin (Ratio {ratio:.2f}, NCC {ncc:.2f})"
            shift = 99.9
        elif ratio < cfg["missing_ratio_thresh_high"] and ncc >= cfg["missing_ncc_thresh_high"] and t_std < cfg["std_thresh_high"]:
            status = "MISSING"
            detail = f"MISSING: Missing Pin (Ratio {ratio:.2f}, NCC {ncc:.2f}, Std {t_std:.1f})"
            shift = 99.9
        elif ratio < cfg["missing_ratio_thresh_mid"] or t_tip is None or ncc < cfg["severe_ncc_thresh"]:
            # Severe pattern degradation, large shift, extreme displacement (ratio < 0.40), or missing tip on present pin -> bent
            if t_tip is None or shift > cfg["severe_shift_thresh"] or ncc < cfg["severe_ncc_thresh"] or ratio < cfg["missing_ratio_thresh_low"]:
                status = "SEVERE_BENT"
                detail = f"SEVERE_BENT: Bent Pin (NCC {ncc:.2f})"
                if t_tip is not None:
                    detail = f"SEVERE_BENT: Bent Pin ({shift:.1f}px, NCC {ncc:.2f})"
            elif shift > cfg["warn_shift_thresh"]:
                status = "WARNING"
                detail = f"WARNING: Bent Pin (NCC {ncc:.2f}, {shift:.1f}px)"
        elif ncc < cfg["warn_ncc_thresh"]:
            # Moderate degradation -> use shift as tiebreaker
            if shift > cfg["moderate_severe_shift_thresh"]:
                status = "SEVERE_BENT"
                detail = f"SEVERE_BENT: Severe Bend ({shift:.1f}px, NCC {ncc:.2f})"
            elif shift > cfg["moderate_warn_shift_thresh"]:
                status = "WARNING"
                detail = f"WARNING: Minor Bend ({shift:.1f}px, NCC {ncc:.2f})"

        pin_results.append({
            "pin": pin, "status": status, "detail": detail,
            "ncc": ncc, "shift": shift, "ratio": ratio,
            "g_tip": g_tip, "t_tip": t_tip
        })

    # Pass 2: Neighbor-aware context escalation
    if cfg["enable_neighbor_escalation"]:
        columns = {}
        for i, pr in enumerate(pin_results):
            col = pr["pin"]["id"].split("_")[0]  # e.g. "C1"
            if col not in columns:
                columns[col] = []
            columns[col].append(i)

        for col, indices in columns.items():
            for idx in indices:
                pr = pin_results[idx]
                if pr["status"] != "PASS":
                    continue
                if pr["ncc"] >= cfg["neighbor_ncc_thresh"] or pr["shift"] <= cfg["neighbor_shift_thresh"]:
                    continue  # high NCC or small shift = definitely straight

                # Check if any immediate neighbor in same column is SEVERE_BENT or MISSING
                row_str = pr["pin"]["id"].split("_")[1]  # e.g. "R3"
                row_num = int(row_str[1:])
                has_defective_neighbor = False
                for other_idx in indices:
                    other_row_str = pin_results[other_idx]["pin"]["id"].split("_")[1]
                    other_row = int(other_row_str[1:])
                    if abs(other_row - row_num) == 1 and pin_results[other_idx]["status"] in ["SEVERE_BENT", "MISSING"]:
                        has_defective_neighbor = True
                        break

                if has_defective_neighbor:
                    pr["status"] = "WARNING"
                    pr["detail"] = f"WARNING: Adjacent Defect (NCC {pr['ncc']:.2f}, neighbor defect)"

    return pin_results

def draw_hud(im_test_aligned, pin_results, stats):
    """Renders the annotated inspection HUD frame panel with separated metrics."""
    h_dim, w_dim = im_test_aligned.shape[:2]
    hud_w = 340
    canvas = np.zeros((h_dim, w_dim + hud_w, 3), dtype=np.uint8)
    canvas[:, :w_dim] = im_test_aligned
    canvas[:, w_dim:] = 24
    cv2.line(canvas, (w_dim, 0), (w_dim, h_dim), (45, 45, 45), 2)

    # Color map for the 4 statuses
    colors = {
        "PASS": (0, 230, 0),          # Green
        "WARNING": (0, 230, 230),      # Yellow
        "SEVERE_BENT": (40, 40, 255),  # Red (BGR)
        "MISSING": (255, 0, 255)       # Magenta (BGR)
    }

    total_passed = sum(1 for r in pin_results if r["status"] == "PASS")
    total_warning = sum(1 for r in pin_results if r["status"] == "WARNING")
    total_severe_bent = sum(1 for r in pin_results if r["status"] == "SEVERE_BENT")
    total_missing = sum(1 for r in pin_results if r["status"] == "MISSING")

    for pr in pin_results:
        pin = pr["pin"]
        pid = pin["id"]
        cx, cy, w, h = pin["cx"], pin["cy"], pin["w"], pin["h"]
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = x1 + w, y1 + h
        col = colors[pr["status"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 1)
        cv2.circle(canvas, (cx, cy), 1, (255, 100, 0), -1)
        if pr["g_tip"] and pr["t_tip"] and pr["status"] != "MISSING":
            gx, gy = int(x1 + pr["g_tip"][0]), int(y1 + pr["g_tip"][1])
            tx, ty = int(x1 + pr["t_tip"][0]), int(y1 + pr["t_tip"][1])
            cv2.circle(canvas, (tx, ty), 1, col, -1)
            cv2.line(canvas, (gx, gy), (tx, ty), (0, 165, 255), 1)
        cv2.putText(canvas, pid, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)

    # Reference overlay circle
    cv2.circle(canvas, (300, 320), 245, (80, 80, 80), 1, cv2.LINE_AA)

    # HUD text rendering helper
    def put(txt, x, y, sc=0.4, c=(240, 240, 240), th=1):
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

    put("LED PIN INSPECTOR", w_dim + 20, 38, 0.72, (255, 255, 255), 2)
    put("PORTAL v6.0 (METRICS-SPLIT)", w_dim + 20, 54, 0.38, (120, 120, 120))
    cv2.line(canvas, (w_dim + 20, 66), (w_dim + 320, 66), (60, 60, 60), 1)

    hy = 92
    put("OVERALL SYSTEM STATUS:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 24
    
    # Calculate overall status (Missing pins do not cause FAIL status)
    if total_severe_bent > 0:
        ss, sc = "FAIL: BENT PINS", (40, 40, 255)
    elif total_warning > 0:
        ss, sc = "WARNING", (0, 230, 230)
    else:
        ss, sc = "PASS", (0, 230, 0)
        
    cv2.rectangle(canvas, (w_dim + 20, hy - 12), (w_dim + 320, hy + 24), (30, 30, 30), -1)
    cv2.rectangle(canvas, (w_dim + 20, hy - 12), (w_dim + 320, hy + 24), sc, 1)
    put(ss, w_dim + 30, hy + 12, 0.58, sc, 2)

    hy += 50
    cv2.line(canvas, (w_dim + 20, hy), (w_dim + 320, hy), (60, 60, 60), 1)
    hy += 20
    put("HOUSING GLOBAL ALIGNMENT:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 22
    
    global_tx = stats.get("tx", 0.0)
    global_ty = stats.get("ty", 0.0)
    global_rotation = stats.get("rotation", 0.0)
    global_scale = stats.get("scale", 1.0)
    
    put(f"Translation: X={global_tx:+.2f}, Y={global_ty:+.2f} px", w_dim + 30, hy)
    hy += 20
    put(f"Rotation:    {global_rotation:+.3f} deg", w_dim + 30, hy)
    hy += 20
    put(f"Scale Ratio: {global_scale:.4f}x", w_dim + 30, hy)

    hy += 34
    cv2.line(canvas, (w_dim + 20, hy), (w_dim + 320, hy), (60, 60, 60), 1)
    hy += 20
    put("INSPECTION METRICS:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 22
    put("Total Pins Scanned:   22", w_dim + 30, hy, c=(255, 255, 255))
    hy += 18
    put(f"Passed (Green):       {total_passed}", w_dim + 30, hy, c=(0, 230, 0))
    hy += 18
    put(f"Warning (Yellow):     {total_warning}", w_dim + 30, hy, c=(0, 230, 230))
    hy += 18
    put(f"Severe Bent (Red):    {total_severe_bent}", w_dim + 30, hy, c=(40, 40, 255))
    hy += 18
    put(f"Missing (Magenta):    {total_missing}", w_dim + 30, hy, c=(255, 0, 255))

    hy += 28
    cv2.line(canvas, (w_dim + 20, hy), (w_dim + 320, hy), (60, 60, 60), 1)
    hy += 20
    put("LEGEND:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 22
    cv2.rectangle(canvas, (w_dim + 30, hy - 11), (w_dim + 45, hy + 1), (0, 230, 0), -1)
    put("Pin Present & Aligned", w_dim + 60, hy, 0.38)
    hy += 20
    cv2.rectangle(canvas, (w_dim + 30, hy - 11), (w_dim + 45, hy + 1), (0, 230, 230), -1)
    put("Minor Bent / Shifted Pin", w_dim + 60, hy, 0.38)
    hy += 20
    cv2.rectangle(canvas, (w_dim + 30, hy - 11), (w_dim + 45, hy + 1), (40, 40, 255), -1)
    put("Severely Bent Pin", w_dim + 60, hy, 0.38)
    hy += 20
    cv2.rectangle(canvas, (w_dim + 30, hy - 11), (w_dim + 45, hy + 1), (255, 0, 255), -1)
    put("Missing Pin", w_dim + 60, hy, 0.38)

    hy += 30
    cv2.line(canvas, (w_dim + 20, hy), (w_dim + 320, hy), (60, 60, 60), 1)
    hy += 20
    put("PORTAL LOGS:", w_dim + 20, hy, 0.42, (180, 180, 180))
    hy += 22
    if total_missing > 0 or total_severe_bent > 0:
        put("FAIL: CONNECTOR FAULTY", w_dim + 25, hy, 0.38, (40, 40, 255))
    else:
        put("ALL SCANS COMPLETED: PASS", w_dim + 25, hy, 0.38, (0, 230, 0))

    return canvas
