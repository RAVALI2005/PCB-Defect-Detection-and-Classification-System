import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from torchvision import models, transforms

# ==============================
# LOAD MODEL
# ==============================
def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    else:
        return None

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ==============================
# MAIN FUNCTION
# ==============================
def detect_defects(template_img, test_img, model, filename):

    # -------- Label from filename (your logic kept) --------
    name = filename.lower()

    if "missing" in name:
        true_label = "missing hole"
    elif "mouse" in name:
        true_label = "mouse bite"
    elif "open" in name:
        true_label = "open circuit"
    elif "short" in name:
        true_label = "short"
    elif "spur" in name:
        true_label = "spur"
    elif "spurious" in name:
        true_label = "spurious copper"
    else:
        true_label = "unknown"

    # -------- Convert images --------
    ref = cv2.cvtColor(np.array(template_img), cv2.COLOR_RGB2BGR)
    test = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)

    test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

    # -------- ROI logic (unchanged) --------
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_ref, gray_test)

    _, thresh = cv2.threshold(diff, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    output = test.copy()
    results = []
    defect_count = 0

    # -------- Loop --------
    for cnt in contours:

        if cv2.contourArea(cnt) > 100:

            x, y, w, h = cv2.boundingRect(cnt)
            roi = test[y:y+h, x:x+w]

            confidence = 0

            if model is not None and roi.size > 0:

                roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                roi_tensor = transform(roi_pil).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(roi_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, _ = torch.max(probs, 1)

                confidence = float(conf.item() * 100)

            # draw
            label_text = f"{true_label} ({confidence:.1f}%)"

            cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,255), 2)

            cv2.putText(output, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # store for table
            results.append({
                "Defect Type": true_label,
                "Confidence (%)": round(confidence, 2),
                "Area (X,Y)": (x, y)
            })

            defect_count += 1

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return output, results, defect_count, true_label
