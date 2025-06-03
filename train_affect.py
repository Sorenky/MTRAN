from torchsampler import ImbalancedDatasetSampler
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.MTRAN import MTRAN
from sam import SAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(5),
            transforms.RandomCrop(224, padding=32)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ]),
    'other': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}

# Hyperparameters
num_epochs = 100
batch_size = 144
learning_rate = 0.00000075
best_test_acc = 0
patience_counter = 0
epoch_counter = 0

model = MTRAN(num_classes=7).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = SAM(model.parameters(), torch.optim.Adam, lr=learning_rate, rho=0.05, adaptive=False, )
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# Load the dataset
train_dataset = datasets.ImageFolder(root='data/affectnet/train', transform=data_transforms['train'])
train_loader = DataLoader(
        train_dataset,
        sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True)
test_dataset = datasets.ImageFolder(root='data/affectnet/val', transform=data_transforms['other'])
test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True)

# Start training
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # optimizer.step()
        optimizer.first_step(zero_grad=True)

        ####################################

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # optimizer.step()
        optimizer.second_step(zero_grad=True)

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_acc = test_correct / test_total

    epoch_counter += 1

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')

    scheduler.step()

    print(f"Epoch {epoch + 1}")
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    print(f"Now best test accuracy is {best_test_acc}")

print(f"Best test Accuracy: {best_test_acc}")