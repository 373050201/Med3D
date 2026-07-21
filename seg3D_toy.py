import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np



# ---------- 超参 ----------
batch_size = 32
epochs = 5
lr = 0.001
D = 8  # 深度（模拟8层切片）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 构造 3D 数据 ----------
# 从 MNIST 中取图，在深度方向复制 D 层，形成 (D, 28, 28) 的 3D 体数据
# 每层稍微偏移一点，模拟真实 3D 数据中病灶在相邻切片间的连续性
transform = transforms.Compose([transforms.ToTensor()])

class MNIST3D(Dataset):
    def __init__(self, train=True):
        self.mnist = datasets.MNIST('./dataset', train=train, transform=transform)
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, _ = self.mnist[idx]  # (1, 28, 28)=(C, H, W)
        # 在深度方向复制 D 层，每层随机轻微偏移（模拟 3D 结构）
        slices = []
        for d in range(D):
            shift_x = np.random.randint(-1, 2)  # 水平偏移 -1,0,1
            shift_y = np.random.randint(-1, 2)  # 垂直偏移
            shifted = torch.roll(img, shifts=(shift_y, shift_x), dims=(1,2))#沿指定维度循环平移对应量
            slices.append(shifted)
        vol = torch.stack(slices, dim=0)  # (D, 1, 28, 28)
        # 交换维度到 (1, D, 28, 28) 方便 Conv3d
        vol = vol.permute(1, 0, 2, 3)     # (1, D, 28, 28)
        return vol

train_dataset = MNIST3D(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------- 3D MiniSeg ----------
# 和 2D 版结构完全一样，只是 Conv2d → Conv3d，ConvTranspose2d → ConvTranspose3d
class MiniSeg3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码：下采样到 (D/4, 7, 7) 即 (2, 7, 7)，通道数 16
        self.enc = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1),   # (D,28,28)→(D/2,14,14)
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),  # (D/2,14,14)→(D/4,7,7)
            nn.ReLU(),
        )
        # 解码：上采样回 (D,28,28)，通道数回到 1
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),  # (D/4,7,7)→(D/2,14,14)
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, padding=1),   # (D/2,14,14)→(D,28,28)
            nn.Sigmoid(),
        )
    def forward(self, x):
        # 输入x: (B, 1, D, 28, 28)
        # 输出x: (B, 1, D, 28, 28)
        x=self.enc(x)
        x=self.dec(x)
        return x

model = MiniSeg3D().to(device)

# ---------- 损失函数 ----------
bce_loss=nn.BCELoss()# 拟合每个像素
def dice_loss(pred, target, smooth=1.0):# 拟合形状
    inter = (pred * target).sum(dim=(2,3,4))#预测和真实都为前景的像素数
    total = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))#预测为前景与真实为前景的像素数之和
    dice=(2*inter + smooth) / (total + smooth)# dice=2×|预测 ∩ 真实| / (|预测| + |真实|)，smooth防止分母=0
    return (1 - dice).mean()# dice loss = 1 - dice

optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------- 训练 ----------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for vols in train_loader:
        vols = vols.to(device)                     # 原始图片：(B, 1, D, 28, 28)
        targets = (vols > 0.3).float()             # gt：白像素=前景，(B, 1, D, 28, 28)
        
        preds = model(vols)                        # pred：(B, 1, D, 28, 28)
        loss = bce_loss(preds, targets) + dice_loss(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 可视化：取 batch 中第一个样本的中间切片
    model.eval()
    with torch.no_grad():
        vol_sample = vols[0:1]                 # (1,1,D,28,28)
        pred_sample = model(vol_sample)        # (1,1,D,28,28)
        gt_sample = targets[0:1]               # (1,1,D,28,28)
        
        # 取出 numpy 数组，shape (D,28,28)
        vol_np = vol_sample[0,0].cpu().numpy()
        gt_np = gt_sample[0,0].cpu().numpy()
        pred_np = pred_sample[0,0].cpu().numpy()
        
        D = vol_np.shape[0]
        # 创建子图：D 行，3 列
        fig, axes = plt.subplots(D, 3, figsize=(9, 2*D))
        for d in range(D):
            axes[d, 0].imshow(vol_np[d], cmap='gray')
            axes[d, 0].set_ylabel(f'Slice {d}')
            axes[d, 0].axis('off')
            if d == 0: axes[d, 0].set_title('Input')
            
            axes[d, 1].imshow(gt_np[d], cmap='gray')
            axes[d, 1].axis('off')
            if d == 0: axes[d, 1].set_title('GT')
            
            axes[d, 2].imshow(pred_np[d], cmap='gray')
            axes[d, 2].axis('off')
            if d == 0: axes[d, 2].set_title('Pred')
        
        plt.tight_layout()
        plt.show()
    
    print(f'Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}')

print("完成！")