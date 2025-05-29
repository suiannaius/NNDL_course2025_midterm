import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import os
import random
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from collections import defaultdict


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=15, type=int, help='Number of epochs to train [default: 200]')
    parser.add_argument('--lr_fc', default=0.001, type=float, help='Learning rate of fully connected layer')
    parser.add_argument('--lr_frozen', default=0.0001, type=float, help='Learning rate of frozen layers')
    parser.add_argument('--use_pretrain', action='store_true', help="steps to warm up")
    args = parser.parse_args()
    return args


def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    return epoch_losses


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc


def plot_loss_curve(loss_list, pretrain, args):
    plt.figure()
    plt.plot(range(1, len(loss_list)+1), loss_list, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"loss_curve_pretrain_{pretrain}_lr1_{args.lr_fc}_lr2_{args.lr_frozen}.png")  # 保存图像
    plt.show()


def save_args(args, filepath='config.json'):
    with open(filepath, 'w') as f:
        json.dump(vars(args), f, indent=4)
        
        
def main():
    args = getArgs()
    args_name = f'./config.json'
    save_args(args, args_name)
    seed = 926
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr_fc = args.lr_fc
    lr_frozen = args.lr_frozen
    num_epochs = args.num_epochs
    pretrained = args.use_pretrain

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    data_root = './data/caltech101/101_ObjectCategories'
    full_dataset = ImageFolder(root=data_root, transform=transform)

    categories = full_dataset.classes
    bg_index = categories.index("BACKGROUND_Google")

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset):
        if label != bg_index:
            class_indices[label].append(idx)

    train_indices = []
    test_indices = []

    for label, indices in class_indices.items():
        if len(indices) <= 30:
            train_indices.extend(indices)
        else:
            random.shuffle(indices)
            train_indices.extend(indices[:30])
            test_indices.extend(indices[30:])

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    

    if pretrained:
        weights = ResNet18_Weights.DEFAULT  # 加载官方预训练权重
    else:
        weights = None  # 随机初始化权重

    model = resnet18(weights=weights)
    # model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    params_frozen = [param for name, param in model.named_parameters() if not name.startswith('fc.')]
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': lr_fc},
        {'params': params_frozen, 'lr': lr_frozen}
    ])

    print("Training on progress...")
    losses = train(model, train_loader, criterion, optimizer, device=device, num_epochs=num_epochs)

    print("Plot loss curve...")
    plot_loss_curve(losses, pretrain=pretrained, args=args)

    print("Evaluating on progress...")
    evaluate(model, test_loader, device=device)
    
    model_save_path = f"resnet18_pretrain_{pretrained}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    

if __name__ == "__main__":
    main()
