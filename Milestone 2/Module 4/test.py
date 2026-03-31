import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =============================
# Paths (IMPORTANT: separate test folder)
# =============================
test_dir = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\image_subtraction_results"

model_path = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\train_efficientnet_results\pcb_defect_model.pth"

results_dir = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\test_results"

os.makedirs(results_dir, exist_ok=True)

# =============================
# Device
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Transform (same as training)
# =============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =============================
# Load Dataset
# =============================
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)

# =============================
# Load Model
# =============================
model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))

model = model.to(device)
model.eval()

print("Model Loaded Successfully\n")

# =============================
# Testing
# =============================
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# =============================
# Accuracy
# =============================
accuracy = (correct / total) * 100
print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

# =============================
# Confusion Matrix
# =============================
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.show()

# =============================
# Save Results
# =============================
with open(os.path.join(results_dir, "test_results.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}%\n")

print("\nResults saved in:", results_dir)
