/**
 * Transformer Block（Transformer块）模块
 * GPT-2的核心构建块，包含：
 * 1. Multi-Head Self-Attention（多头自注意力）
 * 2. Feed Forward Network（前馈网络）
 * 3. Layer Normalization（层归一化）
 * 4. Residual Connections（残差连接）
 * 
 * GPT-2的架构特点：
 * - 使用Pre-LN（Layer Norm在子层之前）
 * - 两个残差连接
 */

import { MultiHeadAttention } from '../core/Attention.js';
import { FeedForward } from '../core/FeedForward.js';
import { LayerNorm } from '../core/LayerNorm.js';
import { matrixAdd } from '../utils/math.js';

/**
 * Transformer Block
 * GPT-2的单个Transformer层
 */
export class TransformerBlock {
  /**
   * 构造函数
   * @param {number} dModel - 模型维度
   * @param {number} numHeads - 注意力头数
   * @param {number} dFF - 前馈网络隐藏层维度
   * @param {number} dropout - Dropout率（可选，本实现暂不包含）
   */
  constructor(dModel, numHeads, dFF, dropout = 0.0) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.dFF = dFF;
    this.dropout = dropout;
    
    // 多头自注意力层
    this.attention = new MultiHeadAttention(dModel, numHeads);
    
    // 前馈网络层
    this.feedForward = new FeedForward(dModel, dFF);
    
    // Layer Normalization层（Pre-LN架构）
    // 在GPT-2中，Layer Norm在子层之前
    this.ln1 = new LayerNorm(dModel); // 注意力层之前的LN
    this.ln2 = new LayerNorm(dModel); // 前馈层之前的LN
  }

  /**
   * 前向传播
   * @param {number[][]} x - 输入矩阵 (seq_len x d_model)
   * @param {boolean} useCausalMask - 是否使用因果掩码
   * @returns {number[][]} 输出矩阵 (seq_len x d_model)
   */
  forward(x, useCausalMask = true) {
    const seqLen = x.length;
    
    // === 第一个子层：Multi-Head Self-Attention ===
    // Pre-LN: 先归一化，再计算注意力
    const normalized1 = this.ln1.forward(x); // LayerNorm现在支持2D输入
    
    // 计算注意力
    const attentionOutput = this.attention.forward(normalized1, useCausalMask);
    
    // 残差连接: x = x + attention(x)
    const residual1 = matrixAdd(x, attentionOutput);
    
    // === 第二个子层：Feed Forward Network ===
    // Pre-LN: 先归一化，再计算前馈
    const normalized2 = this.ln2.forward(residual1);
    
    // 计算前馈网络
    const ffOutput = this.feedForward.forward(normalized2);
    
    // 残差连接: x = x + ff(x)
    const output = matrixAdd(residual1, ffOutput);
    
    return output;
  }

  /**
   * 设置权重（用于加载预训练模型）
   * @param {Object} weights - 权重对象
   */
  setWeights(weights) {
    if (weights.attention) {
      this.attention.setWeights(weights.attention);
    }
    if (weights.feedForward) {
      this.feedForward.setWeights(weights.feedForward);
    }
    if (weights.ln1) {
      this.ln1.setParams(weights.ln1.gamma, weights.ln1.beta);
    }
    if (weights.ln2) {
      this.ln2.setParams(weights.ln2.gamma, weights.ln2.beta);
    }
  }
}

