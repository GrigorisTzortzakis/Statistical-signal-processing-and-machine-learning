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


%matplotlib inline

# batch_size, learning_rate
PARAM_SETS = [
    {"batch_size": 32,  "learning_rate": 0.01},
    {"batch_size": 128, "learning_rate": 0.001}
]

NUM_USERS = 10
LOCAL_EPOCHS = 1
GLOBAL_ROUNDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# CNN model
class CNN(nn.Module):
    """
    Ένα απλό CNN για το MNIST σε ομοσπονδιακό (federated) σενάριο:
      - Δύο μπλοκ Conv+ReLU+MaxPool
      - Flatten
      - Δύο γραμμικά στρώματα (Linear) για την ταξινόμηση
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


#  Partition Training Data (IID) for 10 Users

def iid_partition(dataset, num_users=10):
    """
    Κάθε χρήστης λαμβάνει ~1/num_users του dataset, τυχαία επιλεγμένο.
    """
    num_items = len(dataset) // num_users
    all_indices = torch.randperm(len(dataset))

    dict_users = {}
    start = 0
    for user_id in range(num_users):
        end = start + num_items
        dict_users[user_id] = all_indices[start:end]
        start = end
    return dict_users

user_indices = iid_partition(train_dataset, NUM_USERS)

# Build Local DataLoaders
def get_local_dataloader(dataset, indices, batch_size):
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader


def local_train(model, dataloader, epochs, lr, device):
    """
    Εκπαίδευση 'model' τοπικά για 'epochs' επαναλήψεις σε 'dataloader',
    με SGD(lr=lr). Επιστρέφει (ενημερωμένο_state_dict, μέσο training_loss).
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

            b_size = images.size(0)
            running_loss += loss.item() * b_size
            total_samples += b_size

    avg_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return copy.deepcopy(model.state_dict()), avg_train_loss

def fedavg(global_model, user_updates, user_data_sizes):
    """
    Weighted average των τοπικών ενημερώσεων.
    Επιστρέφει ένα state_dict για το global μοντέλο.
    """
    new_state_dict = copy.deepcopy(global_model.state_dict())
    for key in new_state_dict.keys():
        new_state_dict[key] = 0.0

    total_data_points = sum(user_data_sizes)
    for i, state_dict_i in enumerate(user_updates):
        user_weight = user_data_sizes[i] / total_data_points
        for key in state_dict_i.keys():
            new_state_dict[key] += state_dict_i[key] * user_weight

    return new_state_dict

def evaluate(model, dataloader, device):
    """
    Αξιολόγηση του 'model' στο δοσμένο dataloader. Επιστρέφει (avg_loss, accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# One DataLoader for the entire test set (global evaluation)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


for ps_idx, ps in enumerate(PARAM_SETS):
    print(f"\n\n============================")
    print(f" FEDERATED RUN {ps_idx+1} of {len(PARAM_SETS)} ")
    print(f" batch_size={ps['batch_size']}, learning_rate={ps['learning_rate']}")
    print(f"============================\n")

    
    local_loaders = []
    user_data_sizes = []
    for u in range(NUM_USERS):
        loader_u = get_local_dataloader(train_dataset, user_indices[u], ps["batch_size"])
        local_loaders.append(loader_u)
        user_data_sizes.append(len(user_indices[u]))

    
    global_model = CNN().to(DEVICE)

    
    global_train_losses = []
    global_test_losses  = []
    global_test_accs    = []

    # Federated training loop
    for round_num in range(GLOBAL_ROUNDS):
        print(f"--- Global Round {round_num+1}/{GLOBAL_ROUNDS} ---")

        user_updates = []
        total_local_loss = 0.0
        total_local_samples = 0

        # Broadcast & Local Train
        for u in range(NUM_USERS):
            # Copy global -> local
            local_model = CNN().to(DEVICE)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            # Local training
            updated_params, local_train_loss = local_train(
                local_model,
                local_loaders[u],
                LOCAL_EPOCHS,
                ps["learning_rate"],
                DEVICE
            )
            user_updates.append(updated_params)

            # Weighted sum για train loss
            num_samples_user = user_data_sizes[u]
            total_local_loss += local_train_loss * num_samples_user
            total_local_samples += num_samples_user

        # FedAvg 
        new_global_state = fedavg(global_model, user_updates, user_data_sizes)
        global_model.load_state_dict(new_global_state)

        # Global training loss
        global_train_loss = total_local_loss / total_local_samples

...         # Evaluate στο test set
...         test_loss, test_acc = evaluate(global_model, test_loader, DEVICE)
... 
...         global_train_losses.append(global_train_loss)
...         global_test_losses.append(test_loss)
...         global_test_accs.append(test_acc)
... 
...         print(f"   Global Train Loss: {global_train_loss:.4f}")
...         print(f"   Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
... 
...     # Plot
...     rounds = range(1, GLOBAL_ROUNDS+1)
... 
...     plt.figure(figsize=(12,4))
... 
...     # Training Loss
...     plt.subplot(1,3,1)
...     plt.plot(rounds, global_train_losses, marker='o')
...     plt.title(f"Global Training Loss\n(Batch={ps['batch_size']}, LR={ps['learning_rate']})")
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
