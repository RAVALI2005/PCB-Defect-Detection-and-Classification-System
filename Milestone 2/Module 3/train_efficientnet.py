import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# =============================
# 1. Paths
# =============================
data_dir = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\image_subtraction_results"
results_dir = r"C:\Users\Ravali\Downloads\PCB_DATASET\PCB_DATASET\train_efficientnet_results"

os.makedirs(results_dir, exist_ok=True)

# Create log file
log_file = os.path.join(results_dir, "training_results.txt")
log = open(log_file, "w")

# =============================
# 2. Configuration
# =============================
batch_size = 32
num_classes = 6
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 3. Preprocessing
# =============================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =============================
# 4. Dataset
# =============================
dataset = datasets.ImageFolder(data_dir, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_data,batch_size=batch_size)

# =============================
# 5. Model
# =============================
model = models.efficientnet_b0(weights="DEFAULT")

model.classifier[1] = nn.Linear(model.classifier[1].in_features,num_classes)

model = model.to(device)

# =============================
# 6. Training Setup
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================
# 7. Track Metrics
# =============================
train_losses = []
train_accuracies = []

# =============================
# 8. Training Loop
# =============================
for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images,labels in train_loader:

        images,labels = images.to(device),labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    train_loss = running_loss/len(train_loader)
    train_accuracy = 100*correct/total

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    log_message = (
        f"Epoch {epoch+1}/{epochs}\n"
        f"Training Loss: {train_loss:.4f}\n"
        f"Training Accuracy: {train_accuracy:.2f}%\n\n"
    )

    print(log_message)
    log.write(log_message)

# =============================
# Final Train Accuracy
# =============================
final_train_accuracy = train_accuracies[-1]

final_message = f"Final Training Accuracy: {final_train_accuracy:.2f}%\n"
print(final_message)
log.write(final_message)

# =============================
# 9. Training Accuracy Graph
# =============================
plt.figure()

plt.plot(train_accuracies, marker='o')

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.savefig(os.path.join(results_dir,"training_accuracy.png"))

plt.show()

# =============================
# 10. Training Loss Graph
# =============================
plt.figure()

plt.plot(train_losses, marker='o')

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig(os.path.join(results_dir,"training_loss.png"))

plt.show()

# =============================
# 11. Save Model
# =============================
torch.save(model.state_dict(),
           os.path.join(results_dir,"pcb_defect_model.pth"))

log.write("\nTraining Complete\n")
log.close()

print("\nTraining Complete")
print("Results saved in:",results_dir)
