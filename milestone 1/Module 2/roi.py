import cv2
import numpy as np
import os

# ===============================
# PATHS
# ===============================
reference_path = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\PCB_USED\12.JPG"
test_path = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\images\Spurious_copper\12_spurious_copper_10.jpg"

# ===============================
# EXTRACT DEFECT TYPE AND IMAGE NAME
# ===============================
defect_type = os.path.basename(os.path.dirname(test_path))  # folder name as defect type
defect_type = defect_type.replace('_', ' ')  # Optional: replace underscores with spaces
image_name = os.path.basename(test_path)  # original image name

# Output folder for this defect type
output_folder = os.path.join(r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\roi_results", os.path.basename(os.path.dirname(test_path)))
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===============================
# LOAD IMAGES
# ===============================
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
# FIND DEFECT CONTOURS
# ===============================
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copy original test image for ROI display
roi_img = test.copy()
defect_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 80:  # Only major defects
        x, y, w, h = cv2.boundingRect(cnt)  # get rectangle around defect
        cv2.rectangle(roi_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red rectangle
        defect_count += 1
        label = f"{defect_type}"  # Use defect type as label
        # Draw label with larger font
        cv2.putText(
            roi_img,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX
            1.0,         # font scale
            (0, 0, 255),
            3            # thickness
        )

print("Major Defects Detected:", defect_count)
print("Defect type:", defect_type)

# ===============================
# SAVE OUTPUT IMAGE
# ===============================
output_path = os.path.join(output_folder, image_name)  # same name as input
cv2.imwrite(output_path, roi_img)
print(f"ROI result saved at: {output_path}")

# ===============================
# DISPLAY
# ===============================
display = cv2.resize(roi_img, (800, 600))
cv2.imshow("PCB Defects Highlighted", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
