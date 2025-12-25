/**
 * Feed Forward Network（前馈神经网络）模块
 * Transformer中的位置感知前馈网络
 * 
 * 结构：
 * Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)
 * 其中 d_ff 通常是 d_model 的4倍
 */

import { matrixMultiply, batchGelu, initializeWeights } from '../utils/math.js';

/**
 * Feed Forward Network层
 * GPT-2使用两层全连接网络，中间使用GELU激活函数
 */
export class FeedForward {
  /**
   * 构造函数
   * @param {number} dModel - 模型维度
   * @param {number} dFF - 前馈网络隐藏层维度（通常是dModel的4倍）
   */
  constructor(dModel, dFF) {
    this.dModel = dModel;
    this.dFF = dFF;
    
    // 第一层：将d_model扩展到d_ff
    this.W1 = initializeWeights(dModel, dFF);
    this.b1 = Array(dFF).fill(0);
    
    // 第二层：将d_ff压缩回d_model
    this.W2 = initializeWeights(dFF, dModel);
    this.b2 = Array(dModel).fill(0);
  }

  /**
   * 前向传播
   * @param {number[][]} x - 输入矩阵 (seq_len x d_model)
   * @returns {number[][]} 输出矩阵 (seq_len x d_model)
   */
  forward(x) {
    const seqLen = x.length;
    
    // 第一层：线性变换 + 偏置
    // x: (seq_len x d_model), W1: (d_model x d_ff)
    // 结果: (seq_len x d_ff)
    const linear1 = matrixMultiply(x, this.W1);
    const withBias1 = linear1.map(row => 
      row.map((val, i) => val + this.b1[i])
    );
    
    // GELU激活函数
    const activated = batchGelu(withBias1);
    
    // 第二层：线性变换 + 偏置
    // activated: (seq_len x d_ff), W2: (d_ff x d_model)
    // 结果: (seq_len x d_model)
    const linear2 = matrixMultiply(activated, this.W2);
    const output = linear2.map(row => 
      row.map((val, i) => val + this.b2[i])
    );
    
    return output;
  }

  /**
   * 设置权重（用于加载预训练模型）
   * @param {Object} weights - 权重对象
   */
  setWeights(weights) {
    if (weights.W1) this.W1 = weights.W1;
    if (weights.b1) this.b1 = weights.b1;
    if (weights.W2) this.W2 = weights.W2;
    if (weights.b2) this.b2 = weights.b2;
  }
}

