/**
 * Positional Encoding（位置编码）模块
 * 为序列中的每个位置添加位置信息，因为Transformer没有循环结构
 * 
 * GPT-2使用可学习的位置嵌入，而不是固定的正弦位置编码
 */

import { initializeWeights } from '../utils/math.js';

/**
 * 可学习的位置嵌入（GPT-2使用的方式）
 * 每个位置都有一个可学习的嵌入向量
 */
export class LearnedPositionalEncoding {
  /**
   * 构造函数
   * @param {number} maxSeqLen - 最大序列长度
   * @param {number} dModel - 模型维度
   */
  constructor(maxSeqLen, dModel) {
    this.maxSeqLen = maxSeqLen;
    this.dModel = dModel;
    
    // 为每个位置创建可学习的嵌入向量
    const limit = Math.sqrt(1.0 / dModel);
    this.positionEmbedding = Array(maxSeqLen).fill(0).map(() => 
      Array(dModel).fill(0).map(() => 
        (Math.random() * 2 - 1) * limit
      )
    );
  }

  /**
   * 前向传播
   * @param {number} seqLen - 当前序列长度
   * @returns {number[][]} 位置嵌入矩阵 (seq_len x d_model)
   */
  forward(seqLen) {
    if (seqLen > this.maxSeqLen) {
      throw new Error(`序列长度 ${seqLen} 超过最大长度 ${this.maxSeqLen}`);
    }
    
    // 返回前seqLen个位置的嵌入
    return this.positionEmbedding.slice(0, seqLen).map(pos => [...pos]);
  }

  /**
   * 设置位置嵌入权重（用于加载预训练模型）
   * @param {number[][]} positionEmbedding - 位置嵌入矩阵
   */
  setPositionEmbedding(positionEmbedding) {
    this.positionEmbedding = positionEmbedding;
  }
}

/**
 * 正弦位置编码（Transformer原始论文使用的方式）
 * 固定函数，不需要学习参数
 */
export class SinusoidalPositionalEncoding {
  /**
   * 构造函数
   * @param {number} maxSeqLen - 最大序列长度
   * @param {number} dModel - 模型维度
   */
  constructor(maxSeqLen, dModel) {
    this.maxSeqLen = maxSeqLen;
    this.dModel = dModel;
    
    // 预计算位置编码矩阵
    this.positionEncoding = this.createPositionEncoding(maxSeqLen, dModel);
  }

  /**
   * 创建正弦位置编码
   * PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   * PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   * @param {number} maxLen - 最大长度
   * @param {number} dModel - 模型维度
   * @returns {number[][]} 位置编码矩阵
   */
  createPositionEncoding(maxLen, dModel) {
    const encoding = [];
    
    for (let pos = 0; pos < maxLen; pos++) {
      const posEncoding = [];
      
      for (let i = 0; i < dModel; i++) {
        if (i % 2 === 0) {
          // 偶数维度使用sin
          const divTerm = Math.pow(10000, (2 * (i / 2)) / dModel);
          posEncoding.push(Math.sin(pos / divTerm));
        } else {
          // 奇数维度使用cos
          const divTerm = Math.pow(10000, (2 * ((i - 1) / 2)) / dModel);
          posEncoding.push(Math.cos(pos / divTerm));
        }
      }
      
      encoding.push(posEncoding);
    }
    
    return encoding;
  }

  /**
   * 前向传播
   * @param {number} seqLen - 当前序列长度
   * @returns {number[][]} 位置编码矩阵 (seq_len x d_model)
   */
  forward(seqLen) {
    if (seqLen > this.maxSeqLen) {
      throw new Error(`序列长度 ${seqLen} 超过最大长度 ${this.maxSeqLen}`);
    }
    
    return this.positionEncoding.slice(0, seqLen).map(pos => [...pos]);
  }
}

