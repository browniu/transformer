# GPT-2 Transformer 架构详解

本文档详细说明GPT-2 Transformer的架构设计和实现原理。

## 📐 整体架构

```
输入Token IDs
    ↓
Token Embedding (vocab_size → d_model)
    ↓
+ Positional Encoding (可学习的)
    ↓
[Transformer Block × N]
    ↓
Final Layer Normalization
    ↓
Language Model Head (d_model → vocab_size)
    ↓
输出Logits
```

## 🔧 核心组件详解

### 1. Token Embedding（词嵌入）

**功能**：将离散的token ID转换为连续的向量表示

**实现**：
```javascript
embedding_matrix: (vocab_size × d_model)
token_vector = embedding_matrix[token_id]
```

**参数**：
- `vocabSize`: 词汇表大小（GPT-2为50257）
- `dModel`: 嵌入维度（通常为768）

### 2. Positional Encoding（位置编码）

**功能**：为序列添加位置信息

**GPT-2使用方式**：可学习的位置嵌入（Learned Positional Encoding）

**实现**：
```javascript
position_embedding: (max_seq_len × d_model)
final_embedding = token_embedding + position_embedding
```

**特点**：
- 每个位置有独立的可学习向量
- 与token embedding相加（不是拼接）

### 3. Transformer Block（Transformer块）

GPT-2的核心构建块，包含两个子层：

#### 3.1 Multi-Head Self-Attention（多头自注意力）

**架构**：
```
输入 x (seq_len × d_model)
    ↓
Layer Normalization (Pre-LN)
    ↓
Multi-Head Attention
    ├─ Head 1: Q, K, V → Attention → (seq_len × d_v)
    ├─ Head 2: Q, K, V → Attention → (seq_len × d_v)
    ├─ ...
    └─ Head H: Q, K, V → Attention → (seq_len × d_v)
    ↓
Concatenate Heads → (seq_len × d_model)
    ↓
Output Projection → (seq_len × d_model)
    ↓
+ Residual Connection
```

**注意力计算流程**：
1. **计算Q, K, V矩阵**
   ```
   Q = XW_q  (seq_len × d_k)
   K = XW_k  (seq_len × d_k)
   V = XW_v  (seq_len × d_v)
   ```

2. **计算注意力分数**
   ```
   scores = QK^T / √d_k  (seq_len × seq_len)
   ```

3. **应用因果掩码**（GPT-2关键特性）
   ```
   causal_mask: 下三角为0，上三角为-∞
   masked_scores = scores + causal_mask
   ```

4. **Softmax归一化**
   ```
   attention_weights = softmax(masked_scores)
   ```

5. **加权求和**
   ```
   output = attention_weights × V  (seq_len × d_v)
   ```

**参数**：
- `numHeads`: 注意力头数（GPT-2 Small为12）
- `dK = dV = dModel / numHeads`: 每个头的维度

#### 3.2 Feed Forward Network（前馈网络）

**架构**：
```
输入 x (seq_len × d_model)
    ↓
Layer Normalization (Pre-LN)
    ↓
Linear(d_model → d_ff)
    ↓
GELU Activation
    ↓
Linear(d_ff → d_model)
    ↓
+ Residual Connection
```

**公式**：
```
FFN(x) = GELU(xW1 + b1)W2 + b2
```

**参数**：
- `dFF`: 前馈网络隐藏层维度（通常是dModel的4倍）

**GELU激活函数**：
```
GELU(x) = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
```

### 4. Layer Normalization（层归一化）

**公式**：
```
LN(x) = γ * (x - μ) / (σ + ε) + β
```

**特点**：
- Pre-LN架构：在子层之前进行归一化
- 对特征维度（d_model）进行归一化
- 可学习的缩放参数γ和偏移参数β

### 5. Language Model Head（语言模型头）

**功能**：将隐藏状态映射到词汇表大小的logits

**实现**：
```
logits = hidden_states × W_lm + b_lm
logits: (seq_len × vocab_size)
```

## 🔄 数据流示例

假设输入序列长度为3，d_model=768：

```
输入: [token_1, token_2, token_3]

1. Token Embedding:
   [768维向量_1, 768维向量_2, 768维向量_3]

2. + Positional Encoding:
   [768维向量_1, 768维向量_2, 768维向量_3]

3. Transformer Block 1:
   - Attention: 每个位置关注所有位置（受因果掩码限制）
   - FFN: 位置感知的前馈变换
   输出: [768维向量_1, 768维向量_2, 768维向量_3]

4. ... (重复N次) ...

5. Final Layer Norm:
   [768维向量_1, 768维向量_2, 768维向量_3]

6. Language Model Head:
   [50257维logits_1, 50257维logits_2, 50257维logits_3]
```

## 🎯 关键设计特点

### 1. Pre-LN架构

GPT-2使用Pre-LN（Layer Norm在子层之前），而不是Post-LN：

```
Pre-LN:  x → LN → Attention → +x
Post-LN: x → Attention → LN → +x
```

**优势**：训练更稳定，梯度流动更好

### 2. 因果掩码（Causal Mask）

确保模型只能看到当前位置及之前的信息：

```
掩码矩阵（3×3示例）:
[ 0, -∞, -∞]
[ 0,  0, -∞]
[ 0,  0,  0]
```

### 3. 残差连接

每个子层都有残差连接，帮助梯度流动：

```
output = input + sublayer(LN(input))
```

### 4. 自回归生成

生成过程是逐步的：
1. 给定初始token序列
2. 预测下一个token
3. 将新token加入序列
4. 重复步骤2-3

## 📊 模型规模

### GPT-2 Small（本实现示例）
- **参数数量**: ~163M
- **dModel**: 768
- **numLayers**: 12
- **numHeads**: 12
- **dFF**: 3072

### GPT-2 Medium
- **参数数量**: ~345M
- **dModel**: 1024
- **numLayers**: 24
- **numHeads**: 16
- **dFF**: 4096

### GPT-2 Large
- **参数数量**: ~762M
- **dModel**: 1280
- **numLayers**: 36
- **numHeads**: 20
- **dFF**: 5120

## 🔍 实现细节

### 数值稳定性

1. **Softmax数值稳定性**
   ```javascript
   // 减去最大值防止溢出
   exp(x - max(x)) / sum(exp(x - max(x)))
   ```

2. **Layer Norm的eps**
   ```javascript
   // 防止除零
   std = sqrt(variance + eps)  // eps = 1e-5
   ```

### 权重初始化

使用Xavier初始化：
```javascript
limit = sqrt(6.0 / (fan_in + fan_out))
weight = random(-limit, limit)
```

### 矩阵维度检查

所有矩阵运算都确保维度匹配：
- Q, K: (seq_len × d_k)
- V: (seq_len × d_v)
- Attention Weights: (seq_len × seq_len)
- Output: (seq_len × d_v) → (seq_len × d_model)

## 🎓 学习要点

1. **注意力机制**：理解Q、K、V的作用和计算流程
2. **多头注意力**：为什么需要多个头，如何拼接
3. **因果掩码**：自回归模型的关键
4. **残差连接**：为什么需要，如何实现
5. **Layer Norm**：与Batch Norm的区别
6. **位置编码**：为什么需要，如何添加

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2论文
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) - 可视化讲解

---

**注意**：本实现专注于教学和原理理解，实际应用建议使用成熟的深度学习框架。

