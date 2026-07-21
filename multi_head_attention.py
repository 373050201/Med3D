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
SEQ_LEN = 28 * 28
D_MODEL = 64            # 总嵌入维度，必须能被 NUM_HEADS 整除
NUM_HEADS = 8           # 多头数（从 1 改为 8）
HEAD_DIM = D_MODEL // NUM_HEADS  # 每个头的维度 = 64/8 = 8

# ---------- 数据 ----------
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(
    datasets.MNIST('./dataset', train=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('./dataset', train=False, transform=transform),
    batch_size=BATCH_SIZE, shuffle=False)

# ---------- 多头自注意力模块 ----------
class MultiHeadSelfAttention(nn.Module):
    """多头自注意力 (Multi-Head Self-Attention)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V 的线性投影（无偏置）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影（多头特有的，将拼接后的结果映射回 d_model，实现头间和头内的双重信息交互）
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        输入 x: (B, N, D)
        输出: (B, N, D)
        """
        B, N, D = x.shape

        # 1. 线性投影得到 Q, K, V，形状 (B, N, D)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 拆分成多个头：将最后一维 D 拆成 (num_heads, head_dim)
        #    reshape 到 (B, N, num_heads, head_dim)
        #    然后 transpose 将 head 维移到第 2 维，得到 (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 对每个头独立计算缩放点积注意力
        #    scores = Q @ K^T / sqrt(head_dim)  形状 (B, num_heads, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, num_heads, N, N)
        #    加权聚合 V: out = attn_weights @ V → (B, num_heads, N, head_dim)
        head_outputs = torch.matmul(attn_weights, V)

        # 4. 拼接所有头的输出：先 transpose 回 (B, N, num_heads, head_dim)
        #    再 view 成 (B, N, D)
        #    contiguous()用于transpose后，view前，保证数据在内存连续
        concat = head_outputs.transpose(1, 2).contiguous().view(B, N, D)

        # 5. 输出投影
        out = self.W_o(concat)
        return out

# ---------- 完整的分类模型 ----------
class SelfAttentionClassifier(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, d_model=D_MODEL, num_heads=NUM_HEADS, num_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        # 将单头注意力替换为多头注意力
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, SEQ_LEN, 1)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.attention(x)
        x = x.mean(dim=1)
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

print("训练完成！")