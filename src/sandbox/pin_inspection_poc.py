import cv2
import numpy as np
import os
import sys
from src.core.inspector import align_frame, inspect_frame, draw_hud

# Resolve paths relative to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'data'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'results'))

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_cli_table(rows_data):
    headers = ["Pin ID", "Status", "NCC Sim", "Shift (px)", "Int Ratio", "Inspection Detail"]
    col_widths = [10, 10, 10, 12, 12, 38]
    border = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    header_str = "|" + "|".join([f" {h:<{w}} " for h, w in zip(headers, col_widths)]) + "|"
    print(border)
    print(header_str)
    print(border)
    for row in rows_data:
        row_str = "|" + "|".join([f" {str(val):<{w}} " for val, w in zip(row, col_widths)]) + "|"
        print(row_str)
    print(border)

def inspect_pins():
    print("=" * 70)
    print("      HIGH-PRECISION 22-PIN AUTOMATED INDUSTRIAL INSPECTION SYSTEM")
    print("=" * 70)

    # 1. Load Images
    print("[1/7] Loading inspection source files...")
    im_gold = cv2.imread(os.path.join(DATA_DIR, 'Good Pins.png'))
    im_test = cv2.imread(os.path.join(DATA_DIR, 'defect_1.png'))
    if im_gold is None or im_test is None:
        print(f"Error: Could not load source images from {DATA_DIR}.")
        return
    print(f"      Golden Reference: {im_gold.shape[1]}x{im_gold.shape[0]} px")
    print(f"      Test Target:      {im_test.shape[1]}x{im_test.shape[0]} px")

    # 2. Alignment
    print("[2/7] SIFT + FLANN Homography alignment...")
    H, im_test_aligned, stats = align_frame(im_gold, im_test)
    if H is None:
        print("Error: Alignment failed. Not enough matching keypoints.")
        return

    print(f"      RANSAC Inliers: {stats['inliers']} / {stats['total_matches']}")
    print(f"      Translation: dx={stats['tx']:+.2f}, dy={stats['ty']:+.2f} px")
    print(f"      Rotation:    {stats['rotation']:+.3f} deg")
    print(f"      Scale:       {stats['scale']:.4f}x")

    # 3-4. Grayscale alignment
    print("[3/7] Normalizing intensity levels...")
    print("[4/7] Initializing spatial grid...")
    gray_gold = cv2.cvtColor(im_gold, cv2.COLOR_BGR2GRAY)
    gray_test_aligned = cv2.cvtColor(im_test_aligned, cv2.COLOR_BGR2GRAY)

    # 5-6. Core Inspection (Pass 1 & Pass 2)
    print("[5/7] Pass 1: Computing per-pin metrics...")
    print("[6/7] Pass 2: Neighbor-aware context escalation...")
    pin_results = inspect_frame(gray_gold, gray_test_aligned)

    # 7. Render Visualization & CLI Report
    print("[7/7] Generating HUD dashboard...")
    canvas = draw_hud(im_test_aligned, pin_results, stats)

    output_path = os.path.join(RESULTS_DIR, 'inspection_result.png')
    cv2.imwrite(output_path, canvas)
    print(f"      Saved '{output_path}'")

    total_passed = sum(1 for r in pin_results if r["status"] == "PASS")
    total_warning = sum(1 for r in pin_results if r["status"] == "WARNING")
    total_rejected = sum(1 for r in pin_results if r["status"] == "REJECT")

    table_rows = []
    for pr in pin_results:
        pin = pr["pin"]
        pid = pin["id"]
        ncc_s = f"{pr['ncc']:.3f}"
        sh_s = f"{pr['shift']:.2f}" if pr['shift'] < 100 else "N/A"
        table_rows.append([pid, pr["status"], ncc_s, sh_s, f"{pr['ratio']:.2f}", pr["detail"]])

    print("=" * 70)
    print("                      QUALITY CONTROL CLI REPORT")
    print("=" * 70)
    build_cli_table(table_rows)
    print(f"\nSummary:")
    print(f"  Passed:   {total_passed:>2d} / 22")
    print(f"  Warnings: {total_warning:>2d} / 22")
    print(f"  Rejected: {total_rejected:>2d} / 22")
    print("=" * 70)
    if total_rejected > 0:
        print("  CONNECTOR STATUS: REJECTED")
    elif total_warning > 0:
        print("  CONNECTOR STATUS: WARNING")
    else:
        print("  CONNECTOR STATUS: PASSED")
    print("=" * 70)

if __name__ == '__main__':
    inspect_pins()
