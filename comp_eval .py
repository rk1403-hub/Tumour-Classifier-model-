import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Aggregation class
class Aggregation(nn.Module):
    def __init__(self, linear_nodes=15, attention_nodes=15, dim=0, aggregation_func=None):
        super().__init__()
        self.linear_nodes = linear_nodes
        self.attention_nodes = attention_nodes
        self.dim = dim
        self.aggregation_func = aggregation_func
        self.attention_layer = nn.Sequential(
            nn.Linear(self.linear_nodes, self.attention_nodes),
            nn.Tanh(),
            nn.Linear(self.attention_nodes, 1)
        )

    def forward(self, x, dim=None):
        gate = self.attention_layer(x)
        attention_map = x * gate
        if dim is None:
            dim = self.dim

        if self.aggregation_func is None:
            attention = torch.mean(attention_map, dim=dim)
        else:
            attention = self.aggregation_func(attention_map, dim=dim)

        return attention

# BagModel class
class BagModel(nn.Module):
    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()
        self.prepNN = prepNN
        self.aggregation_func = aggregation_func
        self.afterNN = afterNN

    def forward(self, input):
        if not isinstance(input, tuple):
            input = (input, torch.zeros(input.size()[0]))

        ids = input[1]
        input = input[0]

        if len(ids.shape) == 1:
            ids.resize_(1, len(ids))

        inner_ids = ids[len(ids) - 1]
        device = input.device

        NN_out = self.prepNN(input)

        unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        bags = unique[idx]
        counts = counts[idx]

        output = torch.empty((input.size(0), len(NN_out[0])), device=device)

        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim=0)

        output = self.afterNN(output)

        return output

# Model creation function
def get_model():
    N_NEURONS = 15
    N_CLASSES = 2
    prepNN = torch.nn.Sequential(
        torch.nn.Linear(2048, N_NEURONS),
        torch.nn.ReLU(),
    )

    agg_func = Aggregation(aggregation_func=torch.mean,
                           linear_nodes=N_NEURONS,
                           attention_nodes=N_NEURONS)

    afterNN = torch.nn.Sequential(
        torch.nn.Dropout(0.25),
        torch.nn.Linear(N_NEURONS, N_CLASSES))

    model = BagModel(prepNN, afterNN, agg_func)
    return model

# Dataset class to load images from subfolders and handle oversampling for evaluation
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, oversample=False):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_counts = {0: 0, 1: 0}  # To store counts of each class

        # Look for images in subfolders named "class0" and "class1"
        for label, class_folder in enumerate(["class0", "class1"]):
            class_dir = os.path.join(image_dir, class_folder)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(".png"):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)
                    self.class_counts[label] += 1  # Increment class count

        if oversample:
            # Oversample the minority class
            max_count = max(self.class_counts.values())
            new_image_paths, new_labels = [], []
            for label in [0, 1]:
                class_image_paths = [self.image_paths[i] for i in range(len(self.image_paths)) if self.labels[i] == label]
                class_labels = [label] * len(class_image_paths)
                
                # Oversample by duplicating the images
                if len(class_image_paths) < max_count:
                    oversample_count = max_count - len(class_image_paths)
                    indices = np.random.choice(len(class_image_paths), size=oversample_count, replace=True)
                    class_image_paths.extend([class_image_paths[i] for i in indices])
                    class_labels.extend([label] * oversample_count)

                new_image_paths.extend(class_image_paths)
                new_labels.extend(class_labels)

            self.image_paths = new_image_paths
            self.labels = new_labels

        print(f"Number of class0 images: {self.labels.count(0)}")
        print(f"Number of class1 images: {self.labels.count(1)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image.view(-1), label

# Function to preprocess an image
def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# Training function

def train_model(model, train_loader, device, num_epochs=80, learning_rate=0.000001):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model((images, torch.zeros(images.size(0)).to(device)))
            loss = criterion(outputs, labels)

            # Check for NaNs in loss
            if torch.isnan(loss).any():
                #print(f"NaN detected in loss at epoch {epoch + 1}. Skipping batch.")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training complete.")
    return model


# Evaluation function
def evaluate_model(model, test_loader, device, thresholds=[0.5]):
    model.eval()
    results = {}

    with torch.no_grad():
        all_probs = []
        all_labels = []
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model((images, torch.zeros(images.size(0)).to(device)))

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        for threshold in thresholds:
            predicted = (all_probs >= threshold).astype(int)
            accuracy = accuracy_score(all_labels, predicted)
            precision = precision_score(all_labels, predicted, zero_division=0)
            recall = recall_score(all_labels, predicted, zero_division=0)
            f1 = f1_score(all_labels, predicted, zero_division=0)
            auc = roc_auc_score(all_labels, all_probs)
            cm = confusion_matrix(all_labels, predicted)

            results[threshold] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'confusion_matrix': cm
            }

            print(f"Threshold: {threshold}")
            print(f"Confusion Matrix:\n{cm}")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            print(f"AUC-ROC: {auc:.2f}")
            print()

    return results

if __name__ == "__main__":
    # Paths
    train_image_dir = '/Users/prajwalrk/Desktop/vm_out/synth_dir'
    eval_image_dir = '/Users/prajwalrk/Desktop/Thesis_data/master'
    model_path = '/Users/prajwalrk/Desktop/vm_out/models/comp80.pth'

    # Set up device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model
    model = get_model().to(device)

    # Create the training dataset and DataLoader
    train_transform = get_transform()
    train_dataset = ImageDataset(train_image_dir, transform=train_transform)
    print(f"Total images in training dataset: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Please check the image directory and file types.")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    model = train_model(model, train_loader, device, num_epochs=80, learning_rate=0.00001)

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Create the evaluation dataset and DataLoader with oversampling
    eval_transform = get_transform()
    eval_dataset = ImageDataset(eval_image_dir, transform=eval_transform, oversample=True)
    print(f"Total images in evaluation dataset: {len(eval_dataset)}")
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty. Please check the image directory and file types.")

    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    thresholds = [0.5, 0.55, 0.6, 0.65]
    results = evaluate_model(model, eval_loader, device, thresholds=thresholds)
