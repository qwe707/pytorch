# PyTorch基础教程

## 目录
1. [张量(Tensor)操作](#1-张量tensor操作)
2. [自动梯度(Autograd)](#2-自动梯度autograd)
3. [神经网络模块(nn.Module)](#3-神经网络模块nnmodule)
4. [优化器(Optimizer)](#4-优化器optimizer)
5. [数据加载与处理](#5-数据加载与处理)

---

## 1. 张量(Tensor)操作

### 1.1 什么是张量？
张量是PyTorch中最基础的数据结构，类似于NumPy的数组，但可以在GPU上运行以加速计算。

### 1.2 创建张量
```python
import torch

# 创建未初始化的张量
x = torch.empty(3, 4)  # 3行4列

# 创建随机张量
x = torch.rand(3, 4)   # 均匀分布 [0, 1)
x = torch.randn(3, 4)  # 标准正态分布

# 创建全0或全1张量
x = torch.zeros(3, 4)
x = torch.ones(3, 4)

# 从数据直接创建
x = torch.tensor([1, 2, 3, 4])
x = torch.tensor([[1, 2], [3, 4]])
```

### 1.3 张量运算
```python
# 加法运算
x = torch.ones(2, 3)
y = torch.ones(2, 3)

# 方法1: 使用+运算符
result = x + y

# 方法2: 使用torch.add函数
result = torch.add(x, y)

# 方法3: 原地操作（会修改原张量）
y.add_(x)  # 带下划线的操作都是原地操作

# 其他常用运算
result = x - y      # 减法
result = x * y      # 逐元素乘法
result = x / y      # 逐元素除法
result = x @ y.T    # 矩阵乘法（@运算符）
result = torch.matmul(x, y.T)  # 矩阵乘法（函数形式）
```

### 1.4 张量索引与切片
```python
x = torch.randn(4, 4)

# 索引单个元素
print(x[0, 0])        # 第一行第一列
print(x[0, 0].item()) # 获取Python数值

# 切片操作
print(x[:, 1])        # 所有行的第二列
print(x[0, :])        # 第一行的所有列
print(x[1:3, :])      # 第2-3行的所有列
```

### 1.5 张量形状变换
```python
x = torch.randn(4, 4)

# view: 改变形状（要求内存连续）
y = x.view(16)        # 变成一维，16个元素
z = x.view(-1, 8)     # -1表示自动计算该维度，结果是2x8

# reshape: 功能类似view，但会在必要时复制数据
y = x.reshape(2, 8)

# transpose: 转置
y = x.T               # 简写
y = x.transpose(0, 1) # 交换维度0和1

# squeeze/unsqueeze: 删除/添加维度
x = torch.zeros(1, 2, 1, 3)
y = x.squeeze()       # 删除所有大小为1的维度，结果是2x3
y = x.squeeze(0)      # 只删除第0维，结果是2x1x3
y = x.unsqueeze(1)    # 在位置1添加维度
```

### 1.6 GPU加速
```python
# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    
    # 直接在GPU上创建张量
    x = torch.ones(3, 4, device=device)
    
    # 将CPU张量移到GPU
    y = torch.ones(3, 4)
    y = y.to(device)
    
    # GPU上的运算
    z = x + y
    
    # 将结果移回CPU
    z = z.to("cpu")
```

---

## 2. 自动梯度(Autograd)

### 2.1 什么是自动梯度？
Autograd是PyTorch的自动微分引擎，用于神经网络训练中的反向传播。它能自动计算梯度。

### 2.2 梯度追踪
```python
import torch

# 创建需要梯度的张量
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 进行运算
y = x + 2
z = y * y * 3
out = z.mean()

print(z)                    # z有grad_fn，表示它是运算结果
print(z.grad_fn)           # 显示创建z的运算
```

### 2.3 反向传播
```python
# 计算梯度
out.backward()

# 查看梯度
print(x.grad)  # d(out)/dx

# 注意：backward()只能对标量调用
# 如果不是标量，需要传入gradient参数
```

### 2.4 阻止梯度追踪
```python
# 方法1: .detach()
x = torch.ones(2, 2, requires_grad=True)
y = x.detach()  # y不会追踪梯度

# 方法2: with torch.no_grad()
with torch.no_grad():
    y = x * 2
    # 这个代码块内的运算不会追踪梯度

# 方法3: .requires_grad_()
x.requires_grad_(False)  # 停止追踪
```

### 2.5 梯度清零
```python
# 梯度会累积，所以每次反向传播前要清零
x = torch.ones(2, 2, requires_grad=True)

# 第一次计算
out = (x * 3).sum()
out.backward()
print(x.grad)  # [[3., 3.], [3., 3.]]

# 第二次计算前清零
x.grad.zero_()
out = (x * 5).sum()
out.backward()
print(x.grad)  # [[5., 5.], [5., 5.]]
```

---

## 3. 神经网络模块(nn.Module)

### 3.1 定义神经网络
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(784, 128)  # 全连接层: 784 -> 128
        self.fc2 = nn.Linear(128, 64)   # 128 -> 64
        self.fc3 = nn.Linear(64, 10)    # 64 -> 10
    
    def forward(self, x):
        # 定义前向传播
        x = F.relu(self.fc1(x))         # 激活函数ReLU
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                 # 输出层不加激活
        return x

# 创建网络实例
net = SimpleNet()
print(net)
```

### 3.2 卷积神经网络(CNN)
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, 3, 1)   # 输入1通道，输出32通道，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Dropout层（防止过拟合）
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 最大池化
        x = self.dropout1(x)
        
        x = torch.flatten(x, 1)  # 展平
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output
```

### 3.3 常用网络层
```python
# 全连接层
nn.Linear(in_features, out_features)

# 卷积层
nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

# 池化层
nn.MaxPool2d(kernel_size, stride)
nn.AvgPool2d(kernel_size, stride)

# 归一化层
nn.BatchNorm2d(num_features)
nn.LayerNorm(normalized_shape)

# Dropout层
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)

# 激活函数
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.LeakyReLU()

# 循环层
nn.LSTM(input_size, hidden_size, num_layers)
nn.GRU(input_size, hidden_size, num_layers)
```

---

## 4. 优化器(Optimizer)

### 4.1 常用优化器
```python
import torch.optim as optim

# SGD（随机梯度下降）
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Adam（自适应学习率）
optimizer = optim.Adam(net.parameters(), lr=0.001)

# RMSprop
optimizer = optim.RMSprop(net.parameters(), lr=0.01)

# AdaGrad
optimizer = optim.Adagrad(net.parameters(), lr=0.01)
```

### 4.2 训练循环
```python
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 1. 清零梯度
        optimizer.zero_grad()
        
        # 2. 前向传播
        output = net(data)
        
        # 3. 计算损失
        loss = criterion(output, target)
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 更新参数
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
```

### 4.3 学习率调度
```python
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

# 每隔step_size个epoch，学习率乘以gamma
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# 指数衰减
scheduler = ExponentialLR(optimizer, gamma=0.9)

# 根据指标自适应调整
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10)

# 使用方法
for epoch in range(num_epochs):
    train(...)
    validate(...)
    scheduler.step()  # 更新学习率
```

---

## 5. 数据加载与处理

### 5.1 Dataset和DataLoader
```python
from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集
dataset = CustomDataset(train_data, train_labels)

# 创建数据加载器
train_loader = DataLoader(
    dataset, 
    batch_size=64,      # 批量大小
    shuffle=True,       # 是否打乱
    num_workers=4       # 多进程加载
)

# 使用数据加载器
for batch_data, batch_labels in train_loader:
    # 训练代码
    pass
```

### 5.2 数据变换(Transform)
```python
from torchvision import transforms

# 定义变换
transform = transforms.Compose([
    transforms.Resize(256),                    # 调整大小
    transforms.CenterCrop(224),                # 中心裁剪
    transforms.ToTensor(),                     # 转换为张量
    transforms.Normalize(                      # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 数据增强（用于训练集）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),         # 随机裁剪
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.ColorJitter(                    # 颜色抖动
        brightness=0.4,
        contrast=0.4,
        saturation=0.4
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 5.3 使用预置数据集
```python
from torchvision import datasets

# MNIST数据集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# CIFAR-10数据集
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# ImageNet数据集
train_dataset = datasets.ImageFolder(
    root='./data/train',
    transform=train_transform
)
```

### 5.4 保存和加载模型
```python
# 保存整个模型
torch.save(net, 'model.pth')

# 加载整个模型
net = torch.load('model.pth')

# 只保存模型参数（推荐）
torch.save(net.state_dict(), 'model_params.pth')

# 加载模型参数
net = SimpleNet()
net.load_state_dict(torch.load('model_params.pth'))

# 保存训练状态（用于断点续训）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 恢复训练
checkpoint = torch.load('checkpoint.pth')
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

## 总结

这份教程覆盖了PyTorch的核心概念：
1. **张量操作** - PyTorch的基础数据结构
2. **自动梯度** - 神经网络训练的核心机制
3. **神经网络模块** - 构建模型的方法
4. **优化器** - 参数更新策略
5. **数据处理** - 高效的数据加载方案

掌握这些内容后，你就可以开始构建和训练自己的深度学习模型了！

## 推荐学习路径
1. 先熟悉张量操作和基本运算
2. 理解自动梯度的工作原理
3. 学习构建简单的神经网络
4. 掌握训练循环和优化器使用
5. 学习数据加载和预处理技巧
6. 尝试更复杂的网络结构（CNN、RNN等）
