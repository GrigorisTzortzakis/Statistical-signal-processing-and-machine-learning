Python 3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


%matplotlib inline


# batch_size, learning_rate
PARAM_SETS = [
    {"batch_size": 32, "learning_rate": 0.01},
    {"batch_size": 128, "learning_rate": 0.001}
]

# Number of total epochs 
EPOCHS = 5  


# MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                       
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = 100.0 * correct / total
    return epoch_loss, accuracy

# Load mnist
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


for i, params in enumerate(PARAM_SETS):
    print(f"\n=== Training with batch_size={params['batch_size']}, "
          f"learning_rate={params['learning_rate']} ===")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    # Initialize model, criterion, optimizer
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"])

    train_losses = []
    test_losses = []
    test_accuracies = []

    # Training loop
...     for epoch in range(EPOCHS):
...         train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
...         test_loss, test_acc = evaluate(model, test_loader, criterion, device)
... 
...         train_losses.append(train_loss)
...         test_losses.append(test_loss)
...         test_accuracies.append(test_acc)
... 
...         print(f"Epoch [{epoch+1}/{EPOCHS}], "
...               f"Train Loss: {train_loss:.4f}, "
...               f"Test Loss: {test_loss:.4f}, "
...               f"Test Acc: {test_acc:.2f}%")
... 
...     # Plot
...    
...     plt.figure(figsize=(10, 4))
... 
...     #  training & test loss
...     plt.subplot(1, 2, 1)
...     plt.plot(train_losses, label='Train Loss')
...     plt.plot(test_losses, label='Test Loss')
...     plt.title(f"Loss Curves\n(Batch {params['batch_size']}, LR {params['learning_rate']})")
...     plt.xlabel("Epoch")
...     plt.ylabel("Loss")
...     plt.legend()
... 
...     # test accuracy
...     plt.subplot(1, 2, 2)
...     plt.plot(test_accuracies, label='Test Accuracy')
...     plt.title(f"Accuracy\n(Batch {params['batch_size']}, LR {params['learning_rate']})")
...     plt.xlabel("Epoch")
...     plt.ylabel("Accuracy (%)")
...     plt.legend()
... 
...     plt.tight_layout()
...     plt.show()
