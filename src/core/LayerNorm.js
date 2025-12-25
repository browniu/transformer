/**
 * Layer Normalization（层归一化）模块
 * 对每个样本的特征维度进行归一化，稳定训练过程
 */

import { matrixAdd, matrixElementWiseMultiply } from '../utils/math.js';

/**
 * Layer Normalization层
 * 公式: LN(x) = γ * (x - μ) / (σ + ε) + β
 * 其中 μ 和 σ 是均值和标准差，γ 和 β 是可学习参数
 */
export class LayerNorm {
  /**
   * 构造函数
   * @param {number} dModel - 模型维度（特征维度）
   * @param {number} eps - 防止除零的小常数，默认1e-5
   */
  constructor(dModel, eps = 1e-5) {
    this.dModel = dModel;
    this.eps = eps;
    
    // 可学习参数：缩放参数γ和偏移参数β
    // 初始化为全1和全0
    this.gamma = Array(dModel).fill(1.0);
    this.beta = Array(dModel).fill(0.0);
  }

  /**
   * 计算均值和标准差
   * @param {number[][]} x - 输入矩阵 (batch_size x seq_len x d_model)
   * @returns {Object} 包含均值和标准差的对象
   */
  computeStats(x) {
    const batchSize = x.length;
    const seqLen = x[0].length;
    const means = [];
    const variances = [];

    // 对每个样本的每个时间步计算均值和方差
    for (let b = 0; b < batchSize; b++) {
      const batchMeans = [];
      const batchVars = [];
      
      for (let s = 0; s < seqLen; s++) {
        const features = x[b][s];
        
        // 计算均值
        const mean = features.reduce((sum, val) => sum + val, 0) / this.dModel;
        
        // 计算方差
        const variance = features.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / this.dModel;
        
        batchMeans.push(mean);
        batchVars.push(variance);
      }
      
      means.push(batchMeans);
      variances.push(batchVars);
    }
    
    return { means, variances };
  }

  /**
   * 前向传播
   * 支持2D输入 (seq_len x d_model) 和 3D输入 (batch_size x seq_len x d_model)
   * @param {number[][]|number[][][]} x - 输入矩阵
   * @returns {number[][]|number[][][]} 归一化后的输出（维度与输入相同）
   */
  forward(x) {
    // 判断是2D还是3D输入
    const is2D = Array.isArray(x[0][0]) === false;
    
    if (is2D) {
      // 2D输入: (seq_len x d_model)
      const seqLen = x.length;
      const output = [];
      
      for (let s = 0; s < seqLen; s++) {
        const features = x[s];
        
        // 计算均值和标准差
        const mean = features.reduce((sum, val) => sum + val, 0) / this.dModel;
        const variance = features.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / this.dModel;
        const std = Math.sqrt(variance + this.eps);
        
        // 归一化: (x - μ) / (σ + ε)
        const normalized = features.map(val => (val - mean) / std);
        
        // 应用可学习参数: γ * normalized + β
        const scaled = normalized.map((val, i) => 
          this.gamma[i] * val + this.beta[i]
        );
        
        output.push(scaled);
      }
      
      return output;
    } else {
      // 3D输入: (batch_size x seq_len x d_model)
      const batchSize = x.length;
      const seqLen = x[0].length;
      const { means, variances } = this.computeStats(x);
      
      const output = [];
      
      for (let b = 0; b < batchSize; b++) {
        const batchOutput = [];
        
        for (let s = 0; s < seqLen; s++) {
          const features = x[b][s];
          const mean = means[b][s];
          const std = Math.sqrt(variances[b][s] + this.eps);
          
          // 归一化: (x - μ) / (σ + ε)
          const normalized = features.map(val => (val - mean) / std);
          
          // 应用可学习参数: γ * normalized + β
          const scaled = normalized.map((val, i) => 
            this.gamma[i] * val + this.beta[i]
          );
          
          batchOutput.push(scaled);
        }
        
        output.push(batchOutput);
      }
      
      return output;
    }
  }

  /**
   * 设置参数（用于加载预训练权重）
   * @param {number[]} gamma - 缩放参数
   * @param {number[]} beta - 偏移参数
   */
  setParams(gamma, beta) {
    this.gamma = gamma;
    this.beta = beta;
  }
}

