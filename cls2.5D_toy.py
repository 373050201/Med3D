"""
二分类：判断按深度堆叠的k张MNIST图片是否是同一数字，若一致返回1，否则返回0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import random

# ---------- 0. 设备设置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 1. 加载 MNIST 原始数据 ----------
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./dataset', train=True, transform=transform)
mnist_test = datasets.MNIST(root='./dataset', train=False, transform=transform)

# ---------- 2. 构造“一致性判断”任务的 2.5D 数据集 ----------
def create_consistency_dataset(mnist_dataset, num_samples, k=5, consistency_ratio=0.5):
    data = []
    labels = []
    # 预先按数字缓存样本索引，避免生成每个切片时都完整遍历一次 MNIST 数据集。
    if hasattr(mnist_dataset, 'targets'):
        digit_indices = {
            digit: torch.where(mnist_dataset.targets == digit)[0].tolist()
            for digit in range(10)
        }
    else:
        digit_indices = {digit: [] for digit in range(10)}
        for idx, (_, digit) in enumerate(mnist_dataset):
            digit_indices[int(digit)].append(idx)
    for _ in range(num_samples):
        if random.random() < consistency_ratio:
            # 5 张独立的同类数字，而不是同一张图复制 5 遍
            digit = random.randint(0, 9)
            indices = random.sample(digit_indices[digit], k)
            slices = [mnist_dataset[idx][0] for idx in indices]
            label = 1
        else:
            # 5 个不同数字
            digits = random.sample(range(10), k)
            slices = [
                mnist_dataset[random.choice(digit_indices[digit])][0]
                for digit in digits
            ]
            label = 0
        stack = torch.stack(slices, dim=0)
        data.append(stack)
        labels.append(label)
    data_tensor = torch.stack(data, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return data_tensor, labels_tensor

train_data, train_labels = create_consistency_dataset(mnist_train, num_samples=2000, k=5)
test_data, test_labels = create_consistency_dataset(mnist_test, num_samples=400, k=5)

print(f"训练数据形状: {train_data.shape}")
print(f"训练标签形状: {train_labels.shape}")
print(f"标签分布: 一致={train_labels.sum().item()}, 不一致={len(train_labels)-train_labels.sum().item()}")

# ---------- 3. 定义模型 ----------
class Model2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 2)

    def forward(self, x):
        x = x[:, 2]          # 只取中间切片
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Model2_5D(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.conv1 = nn.Conv2d(k, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 2)

    def forward(self, x):
        x = x.squeeze(2)     # (batch,5,28,28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 实例化模型并移到指定设备
model_2d = Model2D().to(device)
model_2_5d = Model2_5D(k=5).to(device)

print(f"\n2D模型参数量: {sum(p.numel() for p in model_2d.parameters())}")
print(f"2.5D模型参数量: {sum(p.numel() for p in model_2_5d.parameters())}")

# ---------- 4. 训练函数（已适配CUDA）----------
def train_model(model, train_data, train_labels, epochs=10, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # 将batch数据移到GPU
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 2 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model

def evaluate_model(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        # 将整个测试集移到GPU（数据量小，直接移）
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)
        outputs = model(test_data)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == test_labels).float().mean().item()
    return accuracy

# ---------- 5. 训练两个模型 ----------
print("\n训练2D模型...")
model_2d = train_model(model_2d, train_data, train_labels, epochs=10)

print("\n训练2.5D模型...")
model_2_5d = train_model(model_2_5d, train_data, train_labels, epochs=10)

# ---------- 6. 评估并对比 ----------
acc_2d = evaluate_model(model_2d, test_data, test_labels)
acc_2_5d = evaluate_model(model_2_5d, test_data, test_labels)

print("\n" + "="*50)
print("结果对比:")
print(f"2D模型准确率: {acc_2d:.4f}")
print(f"2.5D模型准确率: {acc_2_5d:.4f}")
print("="*50)

if acc_2_5d > acc_2d:
    print("\n✅ 2.5D模型表现更好，证明了2.5D的优势！")
    print("原因：2.5D能看到所有5张切片，判断它们是否一致；")
    print("     而2D只能看到中间那张，无法得知其他切片的信息。")
else:
    print("\n❌ 2.5D优势未体现，可能需要调整任务设计或训练参数。")