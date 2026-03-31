import cv2
import numpy as np
import os

# ===============================
# LOAD IMAGES
# ===============================
reference_path = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\PCB_USED\12.JPG"
test_path = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\images\Spurious_copper\12_spurious_copper_10.jpg"
ref = cv2.imread(reference_path)
test = cv2.imread(test_path)

# Resize test to match reference
test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

# ===============================
# CONVERT TO GRAYSCALE & DIFF
# ===============================
gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
diff = cv2.absdiff(gray_ref, gray_test)

# Threshold to detect defects
_, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# ===============================
# FIND DEFECT CENTROIDS
# ===============================
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
defect_centroids = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 80:  # only major defects
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            defect_centroids.append((cx, cy))

print("Major Defects Detected:", len(defect_centroids))

# ===============================
# CREATE FULLY BLACK OUTPUT
# ===============================
output = np.zeros((ref.shape[0], ref.shape[1], 3), dtype=np.uint8)

# Draw white dots at defect locations
for (cx, cy) in defect_centroids:
    cv2.circle(output, (cx, cy), 20, (255, 255, 255), -1)

# ===============================
# SAVE IMAGE SUBTRACTION RESULT
# ===============================
# Extract defect type and image name
defect_type = os.path.basename(os.path.dirname(test_path))  # e.g., Missing_hole
image_name = os.path.basename(test_path)  # e.g., 01_missing_hole_01.jpg

# Create output folder for this defect type
output_folder = os.path.join(r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\image_subtraction_results", defect_type)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Full path to save the image
output_path = os.path.join(output_folder, image_name)
cv2.imwrite(output_path, output)
print(f"Image subtraction result saved at: {output_path}")

# ===============================
# DISPLAY
# ===============================
display = cv2.resize(output, (800, 600))
cv2.imshow("Defects on Full Black Background", display)
cv2.waitKey(0)
cv2.destroyAllWindows()