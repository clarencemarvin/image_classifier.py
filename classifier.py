from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os

# Network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),
            nn.Flatten(), 
            nn.Linear(256*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, xb):
        return self.network(xb)

# Data augmentation and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load and prepare the data
data_dir = 'data'  # Suppose the dataset is stored under this folder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}

# Split the training data to create a validation set
valid_size = int(0.1 * len(image_datasets['train']))  # 10% of training data
train_size = len(image_datasets['train']) - valid_size

train_dataset, valid_dataset = torch.utils.data.random_split(image_datasets['train'], [train_size, valid_size])

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=128, shuffle=False, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training and testing function
def train_test(model, criterion, optimizer, scheduler, num_epochs=25):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    history = dict()
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_training_loss = 0.0
        running_training_accuracy = 0.0
        total_training_predictions = 0
       
        start_time = time.time()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_training_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_training_predictions += labels.size(0)
            running_training_accuracy += (predicted == labels).sum().item()

        epoch_training_accuracy = running_training_accuracy / total_training_predictions * 100
        epoch_training_loss = running_training_loss / total_training_predictions
        
        train_loss.append(epoch_training_loss)
        train_accuracy.append(epoch_training_accuracy)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        running_val_accuracy = 0.0
        total_val_predictions = 0
        
        with torch.no_grad():
            for data in valid_dataloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_predictions += labels.size(0)
                running_val_accuracy += (predicted == labels).sum().item()

        epoch_val_accuracy = running_val_accuracy / total_val_predictions * 100
        epoch_val_loss = running_val_loss / total_val_predictions
        
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)
        
        # Scheduler Step (based on validation loss)
        scheduler.step(epoch_val_loss)
        
        end_time = time.time()
        
        print(f'Epoch: [{epoch + 1}/{num_epochs}], '
              f'Training Accuracy: {epoch_training_accuracy:.1f}%, Training Loss: {epoch_training_loss:.3f}, '
              f'Validation Accuracy: {epoch_val_accuracy:.1f}%, Validation Loss: {epoch_val_loss:.3f}, '
              f'Time: {end_time - start_time:.2f}s')

    print('Finished Training')
    history['train_loss'] = train_loss
    history['train_accuracy'] = train_accuracy
    history['val_loss'] = val_loss
    history['val_accuracy'] = val_accuracy

    # Test Phase
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on test images: {accuracy:.2f}%')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    classes = ['class_name_0', 'class_name_1', 'class_name_2', 'class_name_3', 'class_name_4', 'class_name_5', 'class_name_6', 'class_name_7', 'class_name_8', 'class_name_9']

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]} : {100 * class_correct[i] / class_total[i]:.2f} %')
    
    return history, accuracy

def display_numerical_summary(history):
    min_train_loss = min(history['train_loss'])
    min_val_loss = min(history['val_loss'])
    max_train_accuracy = max(history['train_accuracy'])
    max_val_accuracy = max(history['val_accuracy'])

    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    final_train_accuracy = history['train_accuracy'][-1]
    final_val_accuracy = history['val_accuracy'][-1]

    print(f"Minimum Training Loss: {min_train_loss:.4f}")
    print(f"Minimum Validation Loss: {min_val_loss:.4f}")
    print(f"Maximum Training Accuracy: {max_train_accuracy:.2f}%")
    print(f"Maximum Validation Accuracy: {max_val_accuracy:.2f}%")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
    print(f"Final Validation Accuracy: {final_val_accuracy:.2f}%")

    improvement_train_loss = 100 * (history['train_loss'][0] - final_train_loss) / history['train_loss'][0]
    improvement_val_loss = 100 * (history['val_loss'][0] - final_val_loss) / history['val_loss'][0]
    print(f"Improvement in Training Loss: {improvement_train_loss:.2f}%")
    print(f"Improvement in Validation Loss: {improvement_val_loss:.2f}%")

def detect_fit_status(history):
    last_epoch = len(history['train_loss']) - 1
    if history['train_loss'][last_epoch] < history['val_loss'][last_epoch]:
        if history['train_accuracy'][last_epoch] > history['val_accuracy'][last_epoch]:
            print("Model may be overfitting: Training loss is lower than validation loss with higher training accuracy.")
    elif history['train_loss'][last_epoch] > history['val_loss'][last_epoch]:
        if history['train_accuracy'][last_epoch] < history['val_accuracy'][last_epoch]:
            print("Model may be underfitting: Validation loss is lower than training loss with higher validation accuracy.")

if __name__ == '__main__':
    end = time.time()
    model_ft = Net().to(device)  # Model initialization
    print(model_ft.network)
    criterion = nn.CrossEntropyLoss()  # Loss function initialization

    optimizer_ft = optim.Adam(model_ft.parameters(), lr=5e-4)  # Reduced initial learning rate
    scheduler = lr_scheduler.OneCycleLR(optimizer_ft, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=25)
    
    history, accuracy = train_test(model_ft, criterion, optimizer_ft, scheduler, num_epochs=20)
    
    print(f"Time required: {time.time() - end:.2f}s")
    display_numerical_summary(history)
    detect_fit_status(history)
