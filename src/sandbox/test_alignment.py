import cv2
import numpy as np
import os

# Resolve paths relative to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'data'))
RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'results'))

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def align_images():
    print("Loading images...")
    im_gold = cv2.imread(os.path.join(DATA_DIR, 'Good Pins.png'))
    im_test = cv2.imread(os.path.join(DATA_DIR, 'defects.png'))

    if im_gold is None or im_test is None:
        print(f"Error: Could not load images from {DATA_DIR}.")
        return

    print("Converting to grayscale...")
    gray_gold = cv2.cvtColor(im_gold, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(im_test, cv2.COLOR_BGR2GRAY)

    print("Detecting SIFT keypoints...")
    sift = cv2.SIFT_create()
    kp_gold, des_gold = sift.detectAndCompute(gray_gold, None)
    kp_test, des_test = sift.detectAndCompute(gray_test, None)

    print(f"Keypoints in Gold: {len(kp_gold)}, Test: {len(kp_test)}")

    print("Matching keypoints...")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_gold, des_test)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use top matches
    good_matches = matches[:min(100, len(matches))]
    print(f"Good matches: {len(good_matches)}")

    pts_gold = np.float32([kp_gold[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_test = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    print("Computing Homography...")
    H, mask = cv2.findHomography(pts_test, pts_gold, cv2.RANSAC, 5.0)

    print("Warping test image...")
    h, w, c = im_gold.shape
    im_test_aligned = cv2.warpPerspective(im_test, H, (w, h))

    # Save aligned image
    cv2.imwrite(os.path.join(RESULTS_DIR, 'aligned_test.png'), im_test_aligned)
    print(f"Aligned test image saved as '{os.path.join(RESULTS_DIR, 'aligned_test.png')}'")

    # Save side-by-side or overlay comparison
    overlay = cv2.addWeighted(im_gold, 0.5, im_test_aligned, 0.5, 0)
    cv2.imwrite(os.path.join(RESULTS_DIR, 'alignment_overlay.png'), overlay)
    print(f"Alignment overlay saved as '{os.path.join(RESULTS_DIR, 'alignment_overlay.png')}'")

if __name__ == '__main__':
    align_images()
