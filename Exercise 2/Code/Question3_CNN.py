Python 3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import copy
import math
from collections import defaultdict

%matplotlib inline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# batch_size, learning_rate
PARAM_SETS = [
    {"batch_size": 32,  "learning_rate": 0.01},
    {"batch_size": 128, "learning_rate": 0.001}
]

NUM_USERS = 10         # 10 clients
LOCAL_EPOCHS = 1       # how many epochs each user trains locally per round
GLOBAL_ROUNDS = 5      # total number of federated rounds

# MLP model
class CNN(nn.Module):
    """
    Απλό CNN:
      1) Δύο διαδοχικά Conv(3x3) + ReLU + MaxPool(2x2)
      2) Flatten
      3) Linear(64*7*7 -> 128) + ReLU
      4) Linear(128 -> 10)
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)               
        x = x.view(x.size(0), -1)          
        x = self.classifier(x)             
        return x

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

# Non‐IID Partition
def non_iid_partition_2_classes(dataset, num_users=10):
    """
    Για κάθε i, ο χρήστης i βλέπει κλάσεις (i) και ((i+1)%10).
    """
    class_indices = defaultdict(list)
    for idx, (image, label) in enumerate(dataset):
        class_indices[label].append(idx)

    dict_users = {}
    for i in range(num_users):
        c1 = i % 10
        c2 = (i+1) % 10
        user_idx_list = class_indices[c1] + class_indices[c2]
        user_idx_list = torch.tensor(user_idx_list)[torch.randperm(len(user_idx_list))]
        dict_users[i] = user_idx_list
    return dict_users

user_indices = non_iid_partition_2_classes(train_dataset, NUM_USERS)

def get_local_dataloader(dataset, indices, batch_size):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader


def local_train(model, dataloader, epochs, lr, device):
    """
    Εκπαίδευση 'model' τοπικά για 'epochs' σε 'dataloader' με SGD(lr=lr).
    Επιστρέφει: (updated_state_dict, average_training_loss)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    total_samples = 0
    running_loss = 0.0

    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs_local = images.size(0)
            running_loss += loss.item() * bs_local
            total_samples += bs_local

    avg_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return copy.deepcopy(model.state_dict()), avg_train_loss

def fedavg(global_model, user_updates, user_data_sizes):
    """
    Weighted average του local state_dicts κάθε χρήστη, με βάση το μέγεθος δεδομένων.
    """
    new_state_dict = copy.deepcopy(global_model.state_dict())
    for key in new_state_dict.keys():
        new_state_dict[key] = 0.0

    total_data_points = sum(user_data_sizes)
    for i, state_dict_i in enumerate(user_updates):
        frac = user_data_sizes[i] / total_data_points
        for key in state_dict_i.keys():
            new_state_dict[key] += state_dict_i[key] * frac

    return new_state_dict

def evaluate(model, dataloader, device):
    """
    Επιστρέφει (avg_loss, accuracy) στο 'dataloader'.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# Test loader 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


for ps_i, ps in enumerate(PARAM_SETS):
    print(f"\n\n==================================")
    print(f" FEDERATED NON-IID RUN {ps_i+1} of {len(PARAM_SETS)} ")
    print(f" batch_size={ps['batch_size']}, learning_rate={ps['learning_rate']}")
    print(f"==================================\n")

    # Local loaders
    local_loaders = []
    user_data_sizes = []
    for u in range(NUM_USERS):
        loader_u = get_local_dataloader(train_dataset, user_indices[u], ps["batch_size"])
        local_loaders.append(loader_u)
        user_data_sizes.append(len(user_indices[u]))

    # Global model 
    global_model = CNN().to(DEVICE)

    global_train_losses = []
    global_test_losses  = []
    global_test_accs    = []

    for round_num in range(GLOBAL_ROUNDS):
        print(f"--- Global Round {round_num+1}/{GLOBAL_ROUNDS} ---")

        user_updates = []
        total_local_loss = 0.0
        total_local_samples = 0

        # Broadcast & Local Train
        for u in range(NUM_USERS):
            local_model = CNN().to(DEVICE)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            updated_params, local_loss = local_train(
                local_model,
                local_loaders[u],
                LOCAL_EPOCHS,
                ps["learning_rate"],
                DEVICE
            )
            user_updates.append(updated_params)

            n_samples = user_data_sizes[u]
            total_local_loss += local_loss * n_samples
            total_local_samples += n_samples

        # FedAvg
        new_global_state = fedavg(global_model, user_updates, user_data_sizes)
        global_model.load_state_dict(new_global_state)

        # Training loss 
...         global_train_loss = total_local_loss / total_local_samples
... 
...         # Evaluate on global test set
...         test_loss, test_acc = evaluate(global_model, test_loader, DEVICE)
... 
...         global_train_losses.append(global_train_loss)
...         global_test_losses.append(test_loss)
...         global_test_accs.append(test_acc)
... 
...         print(f"   Global Train Loss: {global_train_loss:.4f}")
...         print(f"   Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
... 
...     rounds = range(1, GLOBAL_ROUNDS+1)
...     plt.figure(figsize=(12,4))
... 
...     # Global Training Loss
...     plt.subplot(1,3,1)
...     plt.plot(rounds, global_train_losses, marker='o')
...     plt.title(f"Global Training Loss\n(Non-IID: Batch={ps['batch_size']}, LR={ps['learning_rate']})")
...     plt.xlabel("Global Round")
...     plt.ylabel("Train Loss")
... 
...     # Test Loss
...     plt.subplot(1,3,2)
...     plt.plot(rounds, global_test_losses, marker='o', color='green')
...     plt.title("Test Loss")
...     plt.xlabel("Global Round")
...     plt.ylabel("Loss")
... 
...     # Test Accuracy
...     plt.subplot(1,3,3)
...     plt.plot(rounds, global_test_accs, marker='o', color='orange')
...     plt.title("Test Accuracy")
...     plt.xlabel("Global Round")
...     plt.ylabel("Accuracy (%)")
... 
...     plt.tight_layout()
...     plt.show()
