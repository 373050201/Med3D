import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage.transform import resize
import matplotlib.pyplot as plt

# ---------- 超参数 ----------
DATA_ROOT = "./dataset/Hippocampus"
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
TARGET_SHAPE = (32, 48, 32)          # (D, H, W)
VAL_RATIO = 0.2

# ---------- 数据集 ----------
class HippocampusDataset(Dataset):
    def __init__(self, image_dir, label_dir, file_list):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        img = nib.load(os.path.join(self.image_dir, name)).get_fdata().astype(np.float32)
        label = nib.load(os.path.join(self.label_dir, name)).get_fdata().astype(np.uint8)

        # 重采样到统一尺寸
        img = resize(img, TARGET_SHAPE, order=1, mode='reflect', preserve_range=True).astype(np.float32)
        label = resize(label, TARGET_SHAPE, order=0, mode='reflect', preserve_range=True).astype(np.uint8)

        # Z-score 归一化
        img = (img - img.mean()) / (img.std() + 1e-8)

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, D, H, W)
        label_tensor = torch.from_numpy(label).long()                # (D, H, W)
        return img_tensor, label_tensor

# 加载文件列表
image_dir = os.path.join(DATA_ROOT, "imagesTr")
label_dir = os.path.join(DATA_ROOT, "labelsTr")
all_files = sorted(os.listdir(image_dir))
print(f"总样本数: {len(all_files)}")

np.random.seed(42)
np.random.shuffle(all_files)
split = int(len(all_files) * (1 - VAL_RATIO))
train_files, val_files = all_files[:split], all_files[split:]
print(f"训练集长度: {len(train_files)}, 验证集长度: {len(val_files)}")

train_dataset = HippocampusDataset(image_dir, label_dir, train_files)
val_dataset = HippocampusDataset(image_dir, label_dir, val_files)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

# ==================== CTB 模块定义 ====================
# === 新增：交叉注意力模块（单头） ===
class CrossAttention3D(nn.Module):
    """单头交叉注意力，Q来自解码器，K/V来自编码器同层"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model#嵌入维度，即QKV的维度
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, kv):
        """
        q:  解码器特征 (B, C, D, H, W)
        kv: 编码器特征 (B, C, D, H, W)  注意：C必须相同
        输出: (B, C, D, H, W)
        """
        B, C, D, H, W = q.shape
        N = D * H * W  # token数

        # 展平为序列 (B, N, C)
        q_flat = q.reshape(B, C, N).transpose(1, 2)
        kv_flat = kv.reshape(B, C, N).transpose(1, 2)

        Q = self.W_q(q_flat)   # (B, N, C)，C为特征数
        K = self.W_k(kv_flat)
        V = self.W_v(kv_flat)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # (B, N, N)
        attn = torch.softmax(scores, dim=-1) # (B, N, N)
        out_flat = torch.matmul(attn, V)  # (B, N, C)

        # 恢复3D形状
        out = out_flat.transpose(1, 2).reshape(B, C, D, H, W)
        return out

# ==================== 修改后的 Up 模块（带 CTB） ====================
class UpCTB(nn.Module):
    """用 CTB 替代普通跳跃连接的 Up 模块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        # === 新增：交叉注意力模块（d_model = out_ch，因为上采样后和解码器同层特征通道相同） ===
        self.ctb = CrossAttention3D(d_model=out_ch)
        self.conv = DoubleConv(in_ch, out_ch)  # 保持原 DoubleConv 结构，输入通道in_ch = out_ch + CTB输出通道out_ch

    def forward(self, x1, x2):
        """
        x1: 解码器深层特征（如 x4）
        x2: 编码器同层特征（如 x3）
        """
        # 1. 上采样 x1 → 和 x2 尺寸对齐
        x1_up = self.up(x1)

        # 尺寸微调（防止奇数 padding）
        diff = [x2.size(d+2) - x1_up.size(d+2) for d in range(3)]
        x1_up = nn.functional.pad(x1_up, [
            diff[2]//2, diff[2]-diff[2]//2,
            diff[1]//2, diff[1]-diff[1]//2,
            diff[0]//2, diff[0]-diff[0]//2,
        ])

        # 2. CTB：Q=x1_up(解码器), KV=x2(编码器)
        attn_out = self.ctb(q=x1_up, kv=x2)  # (B, out_ch, D, H, W)

        # 3. 残差连接：注意力输出 + 上采样后的解码器特征（保留定位信息）
        x1_fused = x1_up + attn_out

        # 4. 与原跳跃连接保持一致：拼接 x2 和融合后的 x1_fused
        x = torch.cat([x2, x1_fused], dim=1)  # (B, in_ch, D, H, W)

        return self.conv(x)#一起DoubleConv

# ---------- 3D U-Net（使用 UpCTB） ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.mpconv(x)
    
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)#上采样
        self.conv = DoubleConv(in_ch, out_ch)  # 输入通道 = out_ch + 跳跃连接通道out_ch

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 如果尺寸不一致，pad x1 使其与 x2 一致
        diff = [x2.size(d+2) - x1.size(d+2) for d in range(3)]
        x1 = nn.functional.pad(x1, [diff[2]//2, diff[2]-diff[2]//2,
                                     diff[1]//2, diff[1]-diff[1]//2,
                                     diff[0]//2, diff[0]-diff[0]//2])
        x = torch.cat([x2, x1], dim=1)#拼起来
        return self.conv(x)#一起DoubleConv

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=NUM_CLASSES):
        super().__init__()
        self.inc = DoubleConv(in_ch, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        # === 将前两个 Up 替换为 UpCTB ===
        self.up1 = UpCTB(256, 128)
        self.up2 = UpCTB(128, 64)
        self.up3 = Up(64, 32)
        self.outc = nn.Conv3d(32, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

model = UNet3D().to(DEVICE)

# ---------- 损失函数 ----------
def dice_loss(prob, target):
    target_onehot = torch.nn.functional.one_hot(target, NUM_CLASSES)
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
    dice = 0.0
    for c in range(NUM_CLASSES):
        inter = (prob[:, c] * target_onehot[:, c]).sum(dim=(1, 2, 3))
        union = prob[:, c].sum(dim=(1, 2, 3)) + target_onehot[:, c].sum(dim=(1, 2, 3))
        dice_c = (2 * inter + 1) / (union + 1)
        dice += dice_c.mean()
    return 1 - dice / NUM_CLASSES

def combined_loss(logits, target):
    ce = nn.CrossEntropyLoss()(logits, target)
    prob = torch.softmax(logits, dim=1)
    return ce + dice_loss(prob, target)

optimizer = optim.Adam(model.parameters(), LR)

# ---------- 训练 ----------
best_dice = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = combined_loss(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    dice_vals = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            target_onehot = torch.nn.functional.one_hot(labels, NUM_CLASSES).permute(0,4,1,2,3).float()
            for c in range(NUM_CLASSES):
                inter = (probs[:,c] * target_onehot[:,c]).sum()
                union = probs[:,c].sum() + target_onehot[:,c].sum()
                dice_vals.append(((2*inter + 1) / (union + 1)).item())
    val_dice = np.mean(dice_vals)
    print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        print(f"  -> 保存最佳模型 (Dice={val_dice:.4f})")

    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            img, label = val_dataset[0]
            img_batch = img.unsqueeze(0).to(DEVICE)
            logits = model(img_batch)
            pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()[0]
            label_np = label.numpy()
            mid = label_np.shape[0] // 2
            fig, axes = plt.subplots(1, 3, figsize=(12,4))
            axes[0].imshow(label_np[mid], cmap='viridis', vmin=0, vmax=2)
            axes[0].set_title('GT')
            axes[1].imshow(pred[mid], cmap='viridis', vmin=0, vmax=2)
            axes[1].set_title('Pred')
            axes[2].imshow(img[0,mid].cpu(), cmap='gray')
            axes[2].set_title('Input')
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.show()

print(f"训练结束，最佳验证 Dice: {best_dice:.4f}")