# GPT-2 Transformer 架构实现

<div align="center">

![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Node.js](https://img.shields.io/badge/Node.js-14+-brightgreen.svg)

使用纯JavaScript实现的类GPT-2架构的Transformer模型，100%还原其原理。

[English](#) | 中文

</div>

---

## ✨ 特性

- 🎯 **100%原理还原** - 严格按照Transformer和GPT-2论文实现
- 📦 **零依赖** - 纯JavaScript实现，无需任何外部库
- 🧩 **模块化设计** - 清晰的代码结构，易于理解和扩展
- 📚 **详细注释** - 完整的中文注释和文档
- 🎓 **教育友好** - 适合学习和理解Transformer原理

## 📚 项目结构

```
transformer/
├── src/
│   ├── core/                    # 核心组件
│   │   ├── Attention.js        # 多头注意力机制
│   │   ├── Embedding.js        # 词嵌入层
│   │   ├── FeedForward.js      # 前馈神经网络
│   │   ├── LayerNorm.js        # 层归一化
│   │   └── PositionalEncoding.js # 位置编码
│   ├── models/                  # 模型定义
│   │   ├── TransformerBlock.js # Transformer块
│   │   └── GPT2.js             # GPT-2主模型
│   └── utils/                   # 工具函数
│       └── math.js              # 数学运算工具
├── index.js                     # 入口文件
├── example.js                   # 使用示例
└── README.md                    # 说明文档
```

## 🎯 核心特性

### 1. **完整的GPT-2架构**
- ✅ Token Embedding（词嵌入）
- ✅ Learned Positional Encoding（可学习位置编码）
- ✅ Multi-Head Self-Attention（多头自注意力）
- ✅ Feed Forward Network（前馈网络，使用GELU激活）
- ✅ Layer Normalization（层归一化，Pre-LN架构）
- ✅ Residual Connections（残差连接）
- ✅ Language Model Head（语言模型头）

### 2. **100%原理还原**
- 严格按照Transformer和GPT-2论文实现
- 完整的注意力机制计算流程
- 正确的因果掩码（Causal Mask）实现
- 标准的自回归生成流程

### 3. **模块化设计**
- 每个组件都是独立的Class
- 清晰的模块划分和接口
- 易于扩展和修改

### 4. **高代码可读性**
- 详细的中文注释
- 清晰的变量命名
- 完整的JSDoc文档

## 🚀 快速开始

### 安装依赖

本项目使用ES6模块，无需额外依赖。确保你的Node.js版本支持ES6模块（Node.js 14+）。

### 基本使用

```javascript
import { GPT2 } from './index.js';

// 创建模型配置
const config = {
  vocabSize: 50257,      // 词汇表大小
  dModel: 768,           // 模型维度
  numLayers: 12,         // Transformer层数
  numHeads: 12,          // 注意力头数
  dFF: 3072,             // 前馈网络维度
  maxSeqLen: 1024,       // 最大序列长度
};

// 创建模型
const model = new GPT2(config);

// 前向传播
const tokenIds = [15496, 11, 995]; // 输入token IDs
const output = model.forward(tokenIds);

// 生成下一个token
const nextToken = model.generateNextToken(tokenIds, 1.0);

// 生成完整序列
const generated = model.generate([15496], 20, 1.0);
```

### 运行示例

```bash
node example.js
```

## 📖 核心组件说明

### 1. Multi-Head Attention（多头注意力）

实现缩放点积注意力机制：
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**特点：**
- 支持多个注意力头并行计算
- 自动应用因果掩码（Causal Mask）
- 完整的Q、K、V矩阵计算流程

### 2. Feed Forward Network（前馈网络）

两层全连接网络：
```
FFN(x) = GELU(xW1 + b1)W2 + b2
```

**特点：**
- 使用GELU激活函数（GPT-2标准）
- 维度扩展：d_model → d_ff → d_model

### 3. Transformer Block（Transformer块）

GPT-2的核心构建块：
```
x = x + Attention(LN(x))
x = x + FFN(LN(x))
```

**特点：**
- Pre-LN架构（Layer Norm在子层之前）
- 两个残差连接
- 支持因果掩码

### 4. Layer Normalization（层归一化）

对特征维度进行归一化：
```
LN(x) = γ * (x - μ) / (σ + ε) + β
```

**特点：**
- 支持2D和3D输入
- 可学习的缩放和偏移参数

## 🔬 技术细节

### 注意力机制

1. **计算Q、K、V矩阵**
   ```javascript
   Q = XW_q, K = XW_k, V = XW_v
   ```

2. **计算注意力分数**
   ```javascript
   scores = QK^T / √d_k
   ```

3. **应用因果掩码**
   ```javascript
   masked_scores = scores + causal_mask
   ```

4. **Softmax归一化**
   ```javascript
   attention_weights = softmax(masked_scores)
   ```

5. **加权求和**
   ```javascript
   output = attention_weights * V
   ```

### 位置编码

GPT-2使用**可学习的位置嵌入**，而不是固定的正弦位置编码。每个位置都有一个可学习的嵌入向量。

### 生成策略

支持多种生成策略：
- **温度采样**：控制输出的随机性
- **Top-K采样**：只从概率最高的K个token中选择
- **自回归生成**：逐步生成完整序列

## 📊 模型配置

### GPT-2 Small（示例配置）
```javascript
{
  vocabSize: 50257,
  dModel: 768,
  numLayers: 12,
  numHeads: 12,
  dFF: 3072,
  maxSeqLen: 1024
}
```

### GPT-2 Medium
```javascript
{
  vocabSize: 50257,
  dModel: 1024,
  numLayers: 24,
  numHeads: 16,
  dFF: 4096,
  maxSeqLen: 1024
}
```

### GPT-2 Large
```javascript
{
  vocabSize: 50257,
  dModel: 1280,
  numLayers: 36,
  numHeads: 20,
  dFF: 5120,
  maxSeqLen: 1024
}
```

## 🎓 教育价值

本项目特别适合：
- **学习Transformer原理**：完整的实现帮助理解每个组件
- **理解GPT-2架构**：清晰的代码结构展示GPT-2的设计
- **深度学习实践**：从零实现一个完整的语言模型
- **代码学习**：高质量、可读性强的代码示例

## 📝 注意事项

1. **性能**：这是教学实现，使用纯JavaScript，性能不是最优。生产环境建议使用GPU加速库。

2. **内存**：大模型会消耗大量内存，注意序列长度和模型大小的选择。

3. **数值稳定性**：实现中已考虑数值稳定性（如softmax的数值稳定性处理）。

4. **权重初始化**：当前使用Xavier初始化，实际GPT-2使用特定的初始化策略。

## 🔮 未来扩展

- [ ] 添加训练功能（反向传播、优化器）
- [ ] 支持加载预训练权重
- [ ] 添加Tokenizer集成
- [ ] 性能优化（使用WebAssembly或GPU）
- [ ] 添加更多生成策略（Top-p采样、Beam Search等）

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

本项目仅供学习和研究使用。

## 🙏 致谢

- OpenAI GPT-2论文
- Transformer论文（Attention Is All You Need）
- 所有为Transformer架构做出贡献的研究者

---

**注意**：这是一个教学项目，旨在帮助理解Transformer和GPT-2的原理。实际应用建议使用成熟的深度学习框架（如PyTorch、TensorFlow）或专门的JavaScript库。

