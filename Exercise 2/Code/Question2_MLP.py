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

NUM_USERS = 10         # 10 federated clients
LOCAL_EPOCHS = 1       # local epochs per global round
GLOBAL_ROUNDS = 5      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


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
    Return a dict: user_id -> list of image indices (IID partition).
    Each user gets ~1/num_users of the dataset, randomly chosen.
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
    Train 'model' on 'dataloader' for 'epochs' using SGD(lr=lr).
    Return: (updated_state_dict, average_training_loss)
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

            batch_size_local = images.size(0)
            running_loss += loss.item() * batch_size_local
            total_samples += batch_size_local

    avg_train_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return copy.deepcopy(model.state_dict()), avg_train_loss

def fedavg(global_model, user_updates, user_data_sizes):
    """
    Weighted average of user_updates by number of samples.
    Returns a new state_dict for the global model.
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
    Evaluate model on dataloader -> return (avg_loss, accuracy).
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


# Run FedAvg for Each Param Set

for ps_idx, ps in enumerate(PARAM_SETS):
    print(f"\n\n============================")
    print(f" FEDERATED RUN {ps_idx+1} of {len(PARAM_SETS)} ")
    print(f" batch_size={ps['batch_size']}, learning_rate={ps['learning_rate']}")
    print(f"============================\n")

    
    local_loaders = [
        get_local_dataloader(train_dataset, user_indices[u], ps["batch_size"])
        for u in range(NUM_USERS)
    ]
    user_data_sizes = [len(user_indices[u]) for u in range(NUM_USERS)]

    # Initialize a fresh global model
    global_model = MLP().to(DEVICE)

   
    global_train_losses = []
    global_test_losses  = []
    global_test_accs    = []

    # Federated training loop
    for round_num in range(GLOBAL_ROUNDS):
        print(f"--- Global Round {round_num+1}/{GLOBAL_ROUNDS} ---")

        # List of local updates
        user_updates = []
        
        total_local_train_loss = 0.0
        total_local_samples    = 0

        # Broadcast & local train
        for user_id in range(NUM_USERS):
            # Copy global -> local
            local_model = MLP().to(DEVICE)
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            # Train locally
            updated_params, local_train_loss = local_train(
                local_model,
                local_loaders[user_id],
                LOCAL_EPOCHS,
                ps["learning_rate"],
                DEVICE
            )
            user_updates.append(updated_params)

            # Weighted sum for global train loss
            num_samples_user = user_data_sizes[user_id]
            total_local_train_loss += local_train_loss * num_samples_user
            total_local_samples    += num_samples_user

        # Federated Averaging
        new_global_state = fedavg(global_model, user_updates, user_data_sizes)
        global_model.load_state_dict(new_global_state)

        # Compute global training loss 
        global_train_loss = total_local_train_loss / total_local_samples

        # Evaluate on test set
...         test_loss, test_acc = evaluate(global_model, test_loader, DEVICE)
... 
...         # Log
...         global_train_losses.append(global_train_loss)
...         global_test_losses.append(test_loss)
...         global_test_accs.append(test_acc)
... 
...         print(f"   Global Train Loss: {global_train_loss:.4f}")
...         print(f"   Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
... 
...    
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
