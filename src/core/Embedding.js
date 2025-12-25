/**
 * Embedding（词嵌入）模块
 * 将离散的token ID转换为连续的向量表示
 */

import { initializeWeights } from '../utils/math.js';

/**
 * Token Embedding层
 * 将token索引映射到d_model维度的向量
 */
export class TokenEmbedding {
  /**
   * 构造函数
   * @param {number} vocabSize - 词汇表大小
   * @param {number} dModel - 模型维度（嵌入维度）
   */
  constructor(vocabSize, dModel) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;
    
    // 嵌入矩阵：每一行代表一个token的嵌入向量
    // 使用较小的初始化范围
    const limit = Math.sqrt(1.0 / dModel);
    this.embedding = Array(vocabSize).fill(0).map(() => 
      Array(dModel).fill(0).map(() => 
        (Math.random() * 2 - 1) * limit
      )
    );
  }

  /**
   * 前向传播
   * @param {number[]} tokenIds - token ID数组 (seq_len)
   * @returns {number[][]} 嵌入向量矩阵 (seq_len x d_model)
   */
  forward(tokenIds) {
    return tokenIds.map(id => {
      if (id < 0 || id >= this.vocabSize) {
        throw new Error(`Token ID ${id} 超出词汇表范围 [0, ${this.vocabSize})`);
      }
      // 返回对应token的嵌入向量（深拷贝）
      return [...this.embedding[id]];
    });
  }

  /**
   * 设置嵌入权重（用于加载预训练模型）
   * @param {number[][]} embedding - 嵌入矩阵
   */
  setEmbedding(embedding) {
    this.embedding = embedding;
  }
}

