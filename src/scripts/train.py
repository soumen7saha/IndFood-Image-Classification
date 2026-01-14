import os
import torch
from PIL import Image
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class FoodDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i,cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    
    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
input_size = 224

# ImageNet Normalization values
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Simple transforms - just resize and normalize
train_transforms = transforms.Compose([
    transforms.Resize((230, 230)),
    transforms.RandomRotation(15),           # Rotate up to 10 degrees
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Zoom
    transforms.RandomHorizontalFlip(),       # Horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_dataset = FoodDataset(
    data_dir='../../data/train',
    transform=train_transforms
)

val_dataset = FoodDataset(
    data_dir='../../data/val',
    transform=val_transforms
)

total_train_len = len(train_dataset)
total_val_len = len(val_dataset)

# Define the length of the smaller dataset you want to use
subset_train_len = 25000
subset_val_len = 6000

# Define the lengths for the split: [desired_length, remaining_length]
train_lengths = [subset_train_len, total_train_len - subset_train_len]
val_lengths = [subset_val_len, total_val_len - subset_val_len]

# Randomly split the original dataset into two new datasets, You get a list of datasets; take the first one [0]
smaller_train_dataset = random_split(train_dataset, train_lengths)[0]
smaller_val_dataset = random_split(val_dataset, val_lengths)[0]

train_loader = DataLoader(smaller_train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(smaller_val_dataset, batch_size=32, shuffle=False)

classes = list(train_dataset.class_to_idx)
num_classes = len(classes)

class FoodClassifierConvNext(nn.Module):
    def __init__(self, num_classes=131):
        super(FoodClassifierConvNext, self).__init__()

        # load pre-trained ConvNeXT-S
        self.base_model = models.convnext_small(weights='IMAGENET1K_V1')

        # freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # remove original classifier
        self.base_model.classifier = nn.Identity()

        # add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.output_layer = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)
        return x
        

model = FoodClassifierConvNext(num_classes=131)
model.to(device)

criterion = nn.CrossEntropyLoss()

def train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device):
    best_val_accuracy = 0.0  # Initialize variable to track the best validation accuracy

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            checkpoint_path = f'food_cnext_v33_{epoch+1:02d}_{val_acc:.3f}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')


class FoodClassifierConvNext(nn.Module):
    def __init__(self, size_inner=100, droprate=0.2, num_classes=131):
        super(FoodClassifierConvNext, self).__init__()

        # load pre-trained ConvNeXT-S
        self.base_model = models.convnext_small(weights='IMAGENET1K_V1')

        # freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # remove original classifier
        self.base_model.classifier = nn.Identity()

        # add custom layers
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        # add inner layers
        self.inner = nn.Linear(768, size_inner)  # New inner layer
        self.relu = nn.ReLU()
        # add dropout
        self.dropout = nn.Dropout(droprate)
        self.output_layer = nn.Linear(size_inner, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.inner(x)
        x = self.relu(x)
        x = self.dropout(x)  # apply dropout
        x = self.output_layer(x)
        return x


def make_model(
        learning_rate=0.001,
        size_inner=500,
        droprate=0.2
):
    model = FoodClassifierConvNext(
        num_classes=131,
        size_inner=size_inner,
        droprate=droprate
    )
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

dummy_input = torch.randn(1, 3, 224, 224).to(device)

print("Model Training: \n")
num_epochs = 50
for drop_rate in [0.3,]: #0.2, 0.1]:
    print(f"drop_rate : {drop_rate}")
    model, optimizer = make_model(
        learning_rate=0.001,
        size_inner=500,
        droprate=drop_rate
    )
    
    train_and_evaluate(model, optimizer, train_loader, val_loader, criterion, num_epochs, device)
    print()

# Export to ONNX
onnx_path = "food_classifier_convnexts_v2.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    verbose=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
)
