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
train_files, val_files = all_files[:split], all_files[split:]#将训练样本划分训练集和验证集
print(f"训练集长度: {len(train_files)}, 验证集长度: {len(val_files)}")

train_dataset = HippocampusDataset(image_dir, label_dir, train_files)
val_dataset = HippocampusDataset(image_dir, label_dir, val_files)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

# img,label=train_dataset[0]
# print(img.shape)# (C, D, H ,W) = (1, 32, 48, 32)
# print(label.shape)# (D, H, W) = (32, 48, 32)
# exit(0)

# ---------- 3D U-Net ----------
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
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.outc = nn.Conv3d(32, out_ch, 1)

    def forward(self, x):
        #输入x: (B, 1, D, H, W)，灰度图in_ch=1
        #输出x: (B, C, D, H, W)，C即类别数NUM_CLASSES
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)#跳跃连接，下同
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

model = UNet3D().to(DEVICE)

# ---------- 损失函数 ----------
def dice_loss(prob, target):
    """
    多分类 Dice Loss
    prob: 模型输出沿通道softmax后的结果, shape = (B, C, D, H, W), 每个通道是该像素属于某个类别的概率
    target: 真实标签, shape = (B, D, H, W), 每个体素是整数类别 0/1/2
    """
    # 将整数标签 target 转为 one-hot 编码
    # target 原始形状 (B, D, H, W)，值 0,1,2
    # one_hot 后形状 (B, D, H, W, C)，C=NUM_CLASSES=3
    target_onehot = torch.nn.functional.one_hot(target, NUM_CLASSES)
    
    # permute 调整维度顺序： (B, D, H, W, C) -> (B, C, D, H, W)，保证target_onehot的形状和pred一致
    target_onehot = target_onehot.permute(0, 4, 1, 2, 3).float()
    
    dice = 0.0  # 累计每个类别的 Dice 值
    
    # 对每个类别单独计算 Dice
    for c in range(NUM_CLASSES):
        # inter：预测为该类别的概率 * 真实样本的 one-hot值（真实值=1.0计算才有效）
        # 在 D,H,W 三个维度上求和，得到每个样本的交集大小
        # inter 形状 (B,)  每个元素是该样本在该类上的交集和
        inter = (prob[:, c] * target_onehot[:, c]).sum(dim=(1, 2, 3))
        
        # union：预测为该类别的概率和 + 真实为该类别的体素数
        # 注意：这里不是严格的并集，而是 |pred| + |target|，是 Dice 公式的标准写法
        union = prob[:, c].sum(dim=(1, 2, 3)) + target_onehot[:, c].sum(dim=(1, 2, 3))
        
        # 计算该类别的 Dice 系数，加 smooth=1 防止除零
        # (2*inter + 1) / (union + 1)  形状 (B,)
        dice_c = (2 * inter + 1) / (union + 1)
        
        # 对该 batch 中所有样本的 Dice 取平均，累加到总 dice 上
        dice += dice_c.mean()
    
    # 返回 1 - 平均 Dice（即 Dice Loss）
    # 平均 Dice = dice / NUM_CLASSES
    return 1 - dice / NUM_CLASSES

def combined_loss(logits, target):#总损失
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

    # 验证
    model.eval()
    dice_vals = []#列表，存储每幅图像的dice值
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
    val_dice = np.mean(dice_vals)#所有图像dice的均值
    print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f}")

    if val_dice > best_dice:#保存dice最佳的模型
        best_dice = val_dice
        #torch.save(model.state_dict(), "best_unet3d_hippocampus.pth")
        print(f"  -> 保存最佳模型 (Dice={val_dice:.4f})")

    # 每 5 轮可视化
    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            img, label = val_dataset[0]#img: [1, D, H, W], label: [D, H, W]
            img_batch = img.unsqueeze(0).to(DEVICE)#[B, 1, D, H, W], 当然此处B=1
            logits = model(img_batch)#[B, C, D, H, W]
            # 下式：softmax转概率→沿C维度取最大概率索引，得[B, D, H, W]→转numpy→去掉batch维度
            pred = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()[0]#[D, H, W]，值为0, 1, 2，即每个体素的预测标签
            label_np = label.numpy()
            mid = label_np.shape[0] // 2# 获取深度方向中间层的切片索引（用于显示二维切片）
            fig, axes = plt.subplots(1, 3, figsize=(12,4))# 创建一行三列的子图，总宽12英寸，高4英寸
            axes[0].imshow(label_np[mid], cmap='viridis', vmin=0, vmax=2)# 真实标签的中间层切片，颜色映射为viridis，值范围0~2
            axes[0].set_title('GT')
            axes[1].imshow(pred[mid], cmap='viridis', vmin=0, vmax=2) # 预测标签的中间层切片，同样颜色映射和范围
            axes[1].set_title('Pred')
            axes[2].imshow(img[0,mid].cpu(), cmap='gray')# 显示输入图像的中间层切片（灰度图）
            axes[2].set_title('Input')
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            plt.show()

print(f"训练结束，最佳验证 Dice: {best_dice:.4f}")