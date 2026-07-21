import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np



# ---------- 超参 ----------
batch_size = 64
epochs = 5
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 数据 ----------
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(
    datasets.MNIST('./dataset', train=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# ---------- MiniSeg：极简分割网络 ----------
# 总参数量约 8000，只有 3 层卷积 + 1 层上采样
class MiniSeg(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码：下采样到 7x7，通道数增加到 16
        self.enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),   # 28→14
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),  # 14→7
            nn.ReLU(),
        )
        # 解码：上采样回 28x28，通道数回到 1
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # 7→14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),   # 14→28
            nn.Sigmoid(),  #每个像素 输出 0~1 概率
        )
    def forward(self, x):
        x=self.enc(x)
        x=self.dec(x)
        return x

model = MiniSeg().to(device)

# ---------- 损失函数 ----------
def dice_loss(pred, target, smooth=1.0):
    inter = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    return (1 - (2*inter + smooth) / (union + smooth)).mean()

optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------- 训练 ----------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        targets = (imgs > 0.3).float()  # GT：白像素=前景
        
        preds = model(imgs)
        loss = nn.BCELoss()(preds, targets) + dice_loss(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 可视化
    model.eval()
    with torch.no_grad():
        img, _ = next(iter(train_loader))
        img = img[:1].to(device)
        pred = model(img)
        
        # 提取原图和预测图（转为numpy）
        orig = img[0,0].cpu().numpy()
        pred_map = pred[0,0].cpu().numpy()
        
        # 定义浅紫色 (R,G,B)，归一化到0~1
        purple = np.array([0.82, 0.62, 1.0])  # 浅紫
        
        # 将灰度原图扩展为RGB三通道
        orig_rgb = np.stack([orig]*3, axis=-1)
        # 合成遮罩：根据预测值线性混合
        # 预测为背景（概率低）的地方，显示原始灰度图像；预测为前景（概率高）的地方，显示紫色（概率越高紫色越浓）
        overlay = (1 - pred_map[..., None]) * orig_rgb + pred_map[..., None] * purple
        
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title('Input')
        axes[1].imshow((img > 0.3)[0,0].cpu(), cmap='gray')
        axes[1].set_title('GT')
        axes[2].imshow(overlay)  # 显示合成的彩色遮罩图
        axes[2].set_title('Pred (purple mask)')
        for ax in axes: ax.axis('off')
        plt.suptitle(f'Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}')
        plt.show()
    
    print(f'Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}')

print("完成！")