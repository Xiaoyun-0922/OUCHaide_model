import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置参数
IMG_SIZE = (224, 224)  # 图像尺寸
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4  # 假设有4种伤口类型：擦伤、割伤、烧伤、溃疡
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集类
class WoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
        
    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                images.append((img_path, self.class_to_idx[class_name]))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据增强和转换
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 创建CNN模型
class WoundClassifier(nn.Module):
    def __init__(self, num_classes):
        super(WoundClassifier, self).__init__()
        # 使用预训练的ResNet18作为基础模型
        self.base_model = models.resnet18(pretrained=True)
        
        # 冻结基础模型的参数
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 替换最后的全连接层
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

# 加载数据集
def load_datasets(data_dir):
    # 假设数据目录结构为：
    # data_dir/
    #   ├── train/
    #   │   ├── abrasion/
    #   │   ├── cut/
    #   │   ├── burn/
    #   │   └── ulcer/
    #   └── val/
    #       ├── abrasion/
    #       ├── cut/
    #       ├── burn/
    #       └── ulcer/
    
    train_dataset = WoundDataset(
        os.path.join(data_dir, 'train'),
        transform=data_transforms['train']
    )
    
    val_dataset = WoundDataset(
        os.path.join(data_dir, 'val'),
        transform=data_transforms['val']
    )
    
    return train_dataset, val_dataset

# 训练函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_wound_classifier.pth')
    
    return model

# 主函数
def main():
    # 加载数据集
    data_dir = 'path_to_wound_dataset'  # 替换为你的数据集路径
    train_dataset, val_dataset = load_datasets(data_dir)
    
    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    }
    
    # 初始化模型
    model = WoundClassifier(NUM_CLASSES).to(DEVICE)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, EPOCHS)
    
    print('Training complete')

if __name__ == '__main__':
    main()