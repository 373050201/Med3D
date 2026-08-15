import math
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset


# ---------- 训练超参数 ----------
DATA_ROOT = "./dataset/Hippocampus"
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
TARGET_SHAPE = (32, 48, 32)  # (D, H, W)
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

        # 重采样到统一尺寸。
        img = resize(img, TARGET_SHAPE, order=1, mode="reflect", preserve_range=True).astype(np.float32)
        label = resize(label, TARGET_SHAPE, order=0, mode="reflect", preserve_range=True).astype(np.uint8)

        # Z-score 归一化。
        img = (img - img.mean()) / (img.std() + 1e-8)

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # [1, D, H, W]
        label_tensor = torch.from_numpy(label).long()  # [D, H, W]
        return img_tensor, label_tensor


# ==================== 论文基础卷积块 ====================
class DoubleConv(nn.Module):
    """论文中的两层 3x3x3 Conv + InstanceNorm + LeakyReLU。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        # 输入 [B, in_ch, D, H, W] -> 输出 [B, out_ch, D, H, W]
        return self.conv(x)


# ==================== vscSE：体素空间 + 通道重标定 ====================
class VoxelSpatialChannelSE(nn.Module):
    """
    复现论文 Fig.5 的 vscSE：sSE 与 cSE 并行，最后逐元素相加。

    输入/输出 shape 均为 [B, C, D, H, W]。
    论文通道分支的中间通道数为 C/2，因此 reduction 默认取 2。
    """

    def __init__(self, channels, reduction=2):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)

        # sSE：每个体素位置生成一个空间权重 [B, 1, D, H, W]。
        self.spatial_excitation = nn.Conv3d(channels, 1, kernel_size=1, bias=True)

        # cSE：全局池化后学习每个通道的权重 [B, C, 1, 1, 1]。
        self.channel_excitation = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=True),# 3Dconv替换线性层
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        spatial_weight = torch.sigmoid(self.spatial_excitation(x))  # [B, 1, D, H, W]
        x_sse = x * spatial_weight  # 广播到 C 个通道: [B, C, D, H, W]

        channel_weight = self.channel_excitation(x)  # [B, C, 1, 1, 1]
        x_cse = x * channel_weight  # 广播到所有体素: [B, C, D, H, W]

        # 论文公式: X_vscSE = X_sSE + X_cSE。
        return x_sse + x_cse  # [B, C, D, H, W]


class EncoderStage(nn.Module):
    """每个编码 stage：DoubleConv -> vscSE；下采样由 stage 之间的 MaxPool 完成。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.vscse = VoxelSpatialChannelSE(out_ch)

    def forward(self, x):
        x = self.conv(x)
        return self.vscse(x)


# ==================== CTB 辅助层 ====================
class ChannelLayerNorm3D(nn.Module):
    """对每个体素的 C 维做 LayerNorm，保持 [B,C,D,H,W] 排列。"""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # [B,C,D,H,W] -> [B,D,H,W,C] -> LayerNorm(C) -> [B,C,D,H,W]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class CTBPreprocess(nn.Module):
    """
    对应论文 Fig.6 前半部分：LayerNorm -> 2xConv3D -> ReLU
    -> MaxPool -> Voxel Embedding。

    注意：论文没有说明五个尺度如何变成相同空间尺寸。这里使用
    adaptive_max_pool3d 对齐到最深编码层尺寸，这是本复现的明确假设。
    """

    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.norm = ChannelLayerNorm3D(in_ch)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # Voxel embedding 将各 stage 的通道统一为 embed_dim。
        self.voxel_embedding = nn.Conv3d(in_ch, embed_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x, target_size):
        # x: [B, C_i, D_i, H_i, W_i]
        x = self.norm(x)  # [B, C_i, D_i, H_i, W_i]
        x = self.conv(x)  # [B, C_i, D_i, H_i, W_i]
        x = F.adaptive_max_pool3d(x, target_size)  # [B, C_i, Dt, Ht, Wt]
        x = self.voxel_embedding(x)  # [B, E, Dt, Ht, Wt]，将特征维度C_i映射到嵌入维度E
        return x


class ConvolutionalMultiHeadAttention3D(nn.Module):
    """
    论文 CTB 的卷积式 Q/K/V + 双头注意力。

    输入是五个已经对齐的尺度，每个元素 shape 为 [B,E,Dt,Ht,Wt]。
    将五个尺度的 token 串接后共同计算注意力，因此任一 stage 的 Q 都能
    关注所有 stage 的 K/V，实现真正的跨尺度 cross-fusion。
    输出是五个包含了注意力结果的特征图，shape与输入一致
    """

    def __init__(self, embed_dim, num_heads=2, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 使用 3D 卷积而非 Linear 生成 Q/K/V，以保留体素邻域空间信息。
        self.q_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.k_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.v_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False)
        self.out_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _to_heads(self, x):
        # [B,E,D,H,W] -> [B,heads,N,head_dim]，N=D*H*W。
        batch, _, depth, height, width = x.shape
        tokens = depth * height * width
        x = x.flatten(2).transpose(1, 2)  # [B,N,E]
        x = x.reshape(batch, tokens, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [B,heads,N,head_dim]

    def forward(self, scale_features):
        batch, _, depth, height, width = scale_features[0].shape
        tokens_per_scale = depth * height * width

        # 每个尺度分别卷积投影，再在 token 维串接。
        q = torch.cat([self._to_heads(self.q_proj(x)) for x in scale_features], dim=2)
        k = torch.cat([self._to_heads(self.k_proj(x)) for x in scale_features], dim=2)
        v = torch.cat([self._to_heads(self.v_proj(x)) for x in scale_features], dim=2)
        # q/k/v: [B, heads, 5*N, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, heads, 5*N, 5*N]
        attention = self.dropout(torch.softmax(scores, dim=-1))
        fused = torch.matmul(attention, v)  # [B, heads, 5*N, head_dim]

        fused = fused.permute(0, 2, 1, 3).contiguous()
        fused = fused.reshape(batch, 5 * tokens_per_scale, self.embed_dim)  # [B,5*N,E]

        # 将跨尺度注意力结果拆回五个 stage，并恢复为 3D 特征图。
        outputs = []
        for stage_tokens in fused.split(tokens_per_scale, dim=1):
            stage_tokens = stage_tokens.transpose(1, 2).reshape(
                batch, self.embed_dim, depth, height, width
            )
            outputs.append(self.out_proj(stage_tokens))  # [B,E,Dt,Ht,Wt]
        return outputs


class MultiScaleCTB(nn.Module):
    """
    Cross-fusion Transformer Block。

    输入：五个编码 stage 经 vscSE 后的特征。
    输出：五个经过跨尺度注意力增强的特征，用它们替代原始 U-Net 的直接复制 skip。

    论文未披露部分的实现假设：
    1. 五个尺度通过 adaptive max pooling 对齐到 Stage 5 的空间尺寸；
    2. 注意力后用三线性插值恢复到各自尺度；
    3. 用 1x1x1 Conv 恢复原通道数，再与该 stage 的 vscSE 特征残差相加。
    """

    def __init__(self, channels=(32, 64, 128, 256, 320), embed_dim=64, num_heads=2):
        super().__init__()
        self.preprocess = nn.ModuleList([CTBPreprocess(ch, embed_dim) for ch in channels])
        self.attention = ConvolutionalMultiHeadAttention3D(embed_dim, num_heads=num_heads)
        self.restore = nn.ModuleList(
            [nn.Conv3d(embed_dim, ch, kernel_size=1, bias=False) for ch in channels]
        )

    def forward(self, encoder_features):
        if len(encoder_features) != 5:
            raise ValueError("论文 CTB 需要五个编码 stage 的特征")

        # 以 Stage 5 的尺寸作为 CTB 公共 voxel grid
        target_size = encoder_features[-1].shape[2:]  # (Dt,Ht,Wt)
        embedded = [
            block(feature, target_size)
            for block, feature in zip(self.preprocess, encoder_features)
        ]  # 每个元素: [B,E,Dt,Ht,Wt]

        attended = self.attention(embedded)

        # 最终输出ctb_features = original + 处理后的 voxel_tokens
        # 处理后的 voxel_tokens 包含两个层次的残差：
        # 1. 内部残差：在公共小尺寸上，注意力输出 attended 与预处理嵌入 embedded 相加
        # 2. 跨模块残差：将上述结果上采样并恢复通道后，与原始特征 original 相加
        ctb_features = []
        for original, voxel_tokens, projection in zip(encoder_features, attended, self.restore):
            # CTB 内部残差：attention 输出 + voxel embedding嵌入特征。
            stage_index = len(ctb_features)
            voxel_tokens = voxel_tokens + embedded[stage_index]  # [B,E,Dt,Ht,Wt]

            # 恢复到对应 encoder/decoder 层的空间尺寸与通道数。
            voxel_tokens = F.interpolate(
                voxel_tokens,
                size=original.shape[2:],
                mode="trilinear",
                align_corners=False,
            )  # [B,E,D_i,H_i,W_i]
            voxel_tokens = projection(voxel_tokens)  # [B,C_i,D_i,H_i,W_i]

            # 跨模块残差，强化 CTB 特征传播。
            ctb_features.append(original + voxel_tokens)  # [B,C_i,D_i,H_i,W_i]

        return ctb_features


class DecoderStage(nn.Module):
    """转置卷积上采样 -> 拼接 CTB skip -> DoubleConv -> vscSE。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)
        self.vscse = VoxelSpatialChannelSE(out_ch)

    def forward(self, decoder_feature, ctb_skip):
        # decoder_feature: [B,in_ch,D/2,H/2,W/2]
        decoder_feature = self.up(decoder_feature)  # [B,out_ch,D,H,W]

        # 奇数尺寸时显式对齐到 CTB skip；TARGET_SHAPE 下通常无需执行。
        if decoder_feature.shape[2:] != ctb_skip.shape[2:]:
            decoder_feature = F.interpolate(
                decoder_feature,
                size=ctb_skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # 注意：这里只拼接 CTB 输出，不再直接复制原始 encoder feature。
        x = torch.cat([decoder_feature, ctb_skip], dim=1)  # [B,2*out_ch,D,H,W]
        x = self.conv(x)  # [B,out_ch,D,H,W]
        return self.vscse(x)  # [B,out_ch,D,H,W]


# ==================== 带 vscSE 与 CTB 的五级 3D U-Net ====================
class UNet3DCTB(nn.Module):
    """
    分割子网结构复现，不包含论文额外的四分类分支。

    对 TARGET_SHAPE=(32,48,32)，默认数据流为：
      Stage1 [B, 32, 32,48,32]
      Stage2 [B, 64, 16,24,16]
      Stage3 [B,128,  8,12, 8]
      Stage4 [B,256,  4, 6, 4]
      Stage5 [B,320,  2, 3, 2]
    """

    def __init__(self, in_ch=1, out_ch=NUM_CLASSES, ctb_embed_dim=64, ctb_heads=2):
        super().__init__()
        channels = (32, 64, 128, 256, 320)

        self.enc1 = EncoderStage(in_ch, channels[0])
        self.enc2 = EncoderStage(channels[0], channels[1])
        self.enc3 = EncoderStage(channels[1], channels[2])
        self.enc4 = EncoderStage(channels[2], channels[3])
        self.enc5 = EncoderStage(channels[3], channels[4])
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # CTB 同时接收五个 encoder stage，输出替代原始 skip 的五尺度特征。
        self.ctb = MultiScaleCTB(channels, embed_dim=ctb_embed_dim, num_heads=ctb_heads)

        self.dec4 = DecoderStage(channels[4], channels[3])
        self.dec3 = DecoderStage(channels[3], channels[2])
        self.dec2 = DecoderStage(channels[2], channels[1])
        self.dec1 = DecoderStage(channels[1], channels[0])
        self.outc = nn.Conv3d(channels[0], out_ch, kernel_size=1)

    def forward(self, x):
        # 输入 x: [B,1,D,H,W]
        e1 = self.enc1(x)  # [B, 32,D,  H,  W]
        e2 = self.enc2(self.pool(e1))  # [B, 64,D/2,H/2,W/2]
        e3 = self.enc3(self.pool(e2))  # [B,128,D/4,H/4,W/4]
        e4 = self.enc4(self.pool(e3))  # [B,256,D/8,H/8,W/8]
        e5 = self.enc5(self.pool(e4))  # [B,320,D/16,H/16,W/16]

        # 五个 stage 在 CTB 中先对齐，再进行跨尺度双头注意力。
        s1, s2, s3, s4, bottleneck = self.ctb([e1, e2, e3, e4, e5])

        # decoder 只使用 CTB 重构后的 skip，不直接使用 e1~e4。
        d4 = self.dec4(bottleneck, s4)  # [B,256,D/8,H/8,W/8]
        d3 = self.dec3(d4, s3)  # [B,128,D/4,H/4,W/4]
        d2 = self.dec2(d3, s2)  # [B, 64,D/2,H/2,W/2]
        d1 = self.dec1(d2, s1)  # [B, 32,D,  H,  W]
        return self.outc(d1)  # logits: [B,out_ch,D,H,W]


# 保留旧名字，避免其他脚本原先通过 UNet3D() 创建模型时失效。
UNet3D = UNet3DCTB


# ---------- 损失函数：保留原实验的 CrossEntropy + Dice ----------
def dice_loss(prob, target):
    target_onehot = F.one_hot(target, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
    dice = 0.0
    for channel in range(NUM_CLASSES):
        intersection = (prob[:, channel] * target_onehot[:, channel]).sum(dim=(1, 2, 3))
        denominator = (
            prob[:, channel].sum(dim=(1, 2, 3))
            + target_onehot[:, channel].sum(dim=(1, 2, 3))
        )
        dice += ((2 * intersection + 1) / (denominator + 1)).mean()
    return 1 - dice / NUM_CLASSES


def combined_loss(logits, target):
    ce = nn.CrossEntropyLoss()(logits, target)
    return ce + dice_loss(torch.softmax(logits, dim=1), target)


def build_loaders():
    """延迟创建 DataLoader，使网络类可以被安全 import 和单独测试。"""
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
    return train_dataset, val_dataset, train_loader, val_loader


def shape_check(device=DEVICE):
    """最小前向形状检查；不读取数据集。"""
    model = UNet3DCTB().to(device)
    sample = torch.randn(1, 1, *TARGET_SHAPE, device=device)
    with torch.no_grad():
        output = model(sample)
    print(f"形状检查: 输入 {tuple(sample.shape)} -> 输出 {tuple(output.shape)}")
    assert output.shape == (1, NUM_CLASSES, *TARGET_SHAPE)
    return model


def train():
    train_dataset, val_dataset, train_loader, val_loader = build_loaders()
    model = UNet3DCTB().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
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
        dice_values = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                probabilities = torch.softmax(model(imgs), dim=1)
                target_onehot = F.one_hot(labels, NUM_CLASSES).permute(0, 4, 1, 2, 3).float()
                for channel in range(NUM_CLASSES):
                    intersection = (probabilities[:, channel] * target_onehot[:, channel]).sum()
                    denominator = probabilities[:, channel].sum() + target_onehot[:, channel].sum()
                    dice_values.append(((2 * intersection + 1) / (denominator + 1)).item())

        val_dice = np.mean(dice_values)
        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {total_loss / len(train_loader):.4f} | Val Dice: {val_dice:.4f}"
        )
        if val_dice > best_dice:
            best_dice = val_dice
            print(f"  -> 当前最佳模型 (Dice={val_dice:.4f})")

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                img, label = val_dataset[0]
                pred = model(img.unsqueeze(0).to(DEVICE)).softmax(1).argmax(1).cpu().numpy()[0]
                label_np = label.numpy()
                middle = label_np.shape[0] // 2
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(label_np[middle], cmap="viridis", vmin=0, vmax=2)
                axes[0].set_title("GT")
                axes[1].imshow(pred[middle], cmap="viridis", vmin=0, vmax=2)
                axes[1].set_title("Pred")
                axes[2].imshow(img[0, middle].cpu(), cmap="gray")
                axes[2].set_title("Input")
                for axis in axes:
                    axis.axis("off")
                plt.tight_layout()
                plt.show()

    print(f"训练结束，最佳验证 Dice: {best_dice:.4f}")


if __name__ == "__main__":
    # 先验证完整数据流与输出 shape，再开始原有训练流程。
    shape_check()
    train()
