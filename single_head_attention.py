import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# ---------- 超参数 ----------
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自注意力参数
SEQ_LEN = 28 * 28       # 将图像展平成 784 个 token
D_MODEL = 64            # 每个 token 的嵌入维度（Q/K/V 的维度）
NUM_HEADS = 1           # 单头注意力（本 toy 只用一个头）

# ---------- 数据 ----------
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(
    datasets.MNIST('./dataset', train=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('./dataset', train=False, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False)

# ---------- 单头自注意力模块 ----------
class SingleHeadSelfAttention(nn.Module):
    """标准的单头缩放点积自注意力 (Scaled Dot-Product Attention)"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # Q, K, V 的线性投影（无偏置，简化）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        输入 x: (B, N, D)   B=batch, N=序列长度(token数), D=嵌入维度(特征维度)
        输出: (B, N, D)     注意力加权后的特征
        """
        B, N, D = x.shape
        # 计算 Q, K, V
        Q = self.W_q(x)  # (B, N, D)，我该关注什么
        K = self.W_k(x)  # (B, N, D)，我有什么
        V = self.W_v(x)  # (B, N, D)，我实际能提供什么

        # 计算注意力分数：Q × K^T / sqrt(D)
        # Q × K^T 形状 (B, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)

        # Softmax 进一步得到归一化的注意力权重
        attn_weights = torch.softmax(scores, dim=-1)  # (B, N, N)，(i, j)<=>token_i对token_j的注意力权重（已归一化）

        # 加权聚合 V：out = attn_weights × V → (B, N, D)，即按注意力权重重新计算每个token的D个特征值
        # 忽略Batch维度，(i, j)<=>token_i在特征维度j上的新特征值（融合了所有token在特征维度j上的特征）
        out = torch.matmul(attn_weights, V)
        return out

# ---------- 完整的分类模型 ----------
class SelfAttentionClassifier(nn.Module):
    """使用单头自注意力 + 全局平均池化 + 全连接层进行 MNIST 分类"""
    def __init__(self, seq_len=SEQ_LEN, d_model=D_MODEL, num_classes=10):
        super().__init__()
        # 将每个像素值 (1维) 投影到 d_model 维
        self.input_proj = nn.Linear(1, d_model)
        # 可学习的位置编码 [seq_len, d_model]
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        # 单头自注意力
        self.attention = SingleHeadSelfAttention(d_model)
        # 分类头：全局平均池化 + 全连接
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),   # 稳定训练
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        输入 x: (B, 1, 28, 28)  MNIST 原始图像
        输出 logits: (B, 10)
        """
        B = x.shape[0]
        # 展平为 (B, 784, 1)  每个像素作为一个 token，特征维度为1
        x = x.view(B, SEQ_LEN, 1)
        # 投影到 d_model 维: (B, 784, d_model)
        x = self.input_proj(x)
        # 加上位置编码
        x = x + self.pos_embedding
        # 自注意力: (B, 784, d_model)
        x = self.attention(x)
        # 全局平均池化: 在 token 维度上求平均，得到 (B, d_model)
        x = x.mean(dim=1)
        # 分类头: (B, 10)
        logits = self.classifier(x)
        return logits

model = SelfAttentionClassifier().to(DEVICE)

# ---------- 训练 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")

print("完成！")