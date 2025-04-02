import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score , recall_score , f1_score , roc_auc_score

# Constants
N_CLASSES = 2
N_NEURONS = 15
DATA_SIZE = 2048  # Adjusted to 2048 to match the pretrained model
BATCH_SIZE = 32

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

# Model creation function with input size adjusted to 2048
def get_model():
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

# Dataset class to load images from subfolders
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
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

        # Print the number of images for each class
        print(f"Number of class0 images: {self.class_counts[0]}")
        print(f"Number of class1 images: {self.class_counts[1]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image.view(-1), label


# Function to preprocess an image (adjusted for 2048 input size)
def get_transform():
    return transforms.Compose([
        transforms.Resize((64, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# Evaluation function with probability-based classification and confusion matrix
def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    predictions, actuals = [], []
    predicted_probs = []  # To store the predicted probabilities for AUC-ROC

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model((images, torch.zeros(images.size(0)).to(device)))
            
            probs = torch.softmax(outputs, dim=1)
            predicted_probs.extend(probs[:, 1].cpu().numpy())  # Save the probabilities for class 1
            predicted = (probs[:, 1] >= threshold).long()
            
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    auc_roc = roc_auc_score(actuals, predicted_probs)
    
    cm = confusion_matrix(actuals, predictions)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")
    
    return accuracy, precision, recall, f1, auc_roc

if __name__ == "__main__":
    # Load the pretrained model
    model = get_model()
    pretrained_weights = torch.load("/Users/prajwalrk/Desktop/code_thesis/model/PretrainedModel#1.pth", map_location='cpu')
    model.load_state_dict(pretrained_weights)

    # Load images from folder
    image_dir = "/Users/prajwalrk/Desktop/Thesis_data/master"

    # Create the dataset and DataLoader
    transform = get_transform()
    dataset = ImageDataset(image_dir, transform=transform)
    print(f"Total images in dataset: {len(dataset)}")
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check the image directory and file types.")

    # Create test DataLoader using the entire dataset
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Set up device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate the model using the complete dataset with threshold 0.5
    accuracy, precision, recall, f1, auc_roc = evaluate_model(model, test_loader, device, threshold=0.5)
    
    # Experiment with different thresholds (e.g., 0.6)
    accuracy, precision, recall, f1, auc_roc = evaluate_model(model, test_loader, device, threshold=0.6)