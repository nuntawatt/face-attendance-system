import numpy as np
import cv2

# ArcFace alignment template
_ARCFACE_REF = np.array([
    [38.2946, 51.6963],   # Point 0: viewer's left eye (person's right eye)
    [73.5318, 51.5014],   # Point 1: viewer's right eye (person's left eye)
    [56.0252, 71.7366],   # Point 2: nose tip
    [41.5493, 92.3655],   # Point 3: viewer's left mouth corner (person's right mouth)
    [70.7299, 92.2041],   # Point 4: viewer's right mouth corner (person's left mouth)
], dtype=np.float32)

# Simulate standard frontal face landmarks
# Let's say a face is at the center of a 112x112 image
right_eye = [40.0, 50.0]  # viewer's left
left_eye = [72.0, 50.0]   # viewer's right
nose = [56.0, 70.0]
right_mouth = [42.0, 90.0]  # viewer's left
left_mouth = [70.0, 90.0]   # viewer's right

# YuNet returns landmarks in order: right_eye (viewer's left), left_eye (viewer's right), nose, right_mouth (viewer's left), left_mouth (viewer's right)
yunet_landmarks = np.array([right_eye, left_eye, nose, right_mouth, left_mouth], dtype=np.float32)

print("=== Landmark coordinates comparison ===")
print("YuNet landmarks (0=right_eye, 1=left_eye, 2=nose, 3=right_mouth, 4=left_mouth):")
for i, pt in enumerate(yunet_landmarks):
    print(f"  Point {i}: {pt}")

print("\nArcFace reference template coordinates:")
for i, pt in enumerate(_ARCFACE_REF):
    print(f"  Point {i}: {pt}")

# Calculate transformation with swapped landmarks:
src_pts_swapped = yunet_landmarks[[1, 0, 2, 4, 3]]
M_swapped, _ = cv2.estimateAffinePartial2D(src_pts_swapped, _ARCFACE_REF)

# Calculate transformation with direct landmarks:
src_pts_direct = yunet_landmarks
M_direct, _ = cv2.estimateAffinePartial2D(src_pts_direct, _ARCFACE_REF)

print("\n=== Transformation Matrix Analysis ===")
print("Swapped mapping M:")
print(M_swapped)
print("Direct mapping M:")
print(M_direct)

# Test warping on a point
test_pt = np.array([40.0, 50.0, 1.0])
warped_swapped = M_swapped @ test_pt
warped_direct = M_direct @ test_pt
print(f"\nWarping point {test_pt[:2]} (Right eye / viewer's left):")
print(f"  Swapped result (should map near [38.3, 51.7]): {warped_swapped}")
print(f"  Direct result  (should map near [38.3, 51.7]): {warped_direct}")

test_pt_left = np.array([72.0, 50.0, 1.0])
warped_swapped_left = M_swapped @ test_pt_left
warped_direct_left = M_direct @ test_pt_left
print(f"\nWarping point {test_pt_left[:2]} (Left eye / viewer's right):")
print(f"  Swapped result (should map near [73.5, 51.5]): {warped_swapped_left}")
print(f"  Direct result  (should map near [73.5, 51.5]): {warped_direct_left}")
