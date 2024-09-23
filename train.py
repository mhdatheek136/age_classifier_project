import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from model import AgeClassifier

# Hyperparameters
num_epochs = 10
batch_size = 35
learning_rate = 0.01

# Initialize DataLoader, Model, Loss Function, Optimizer
dataloader = get_dataloader(batch_size)
model = AgeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()  
    total_correct = 0
    total_samples = 0
    total_loss = 0.0  

    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        total_loss += loss.item()  

    accuracy = total_correct / total_samples  # Calculate accuracy for the epoch
    average_loss = total_loss / len(dataloader)  # Calculate average loss for the epoch

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

