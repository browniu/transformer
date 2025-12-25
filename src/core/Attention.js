/**
 * Multi-Head Attention（多头注意力）模块
 * Transformer的核心组件，实现自注意力机制
 * 
 * 原理：
 * 1. 将输入分为Q(Query)、K(Key)、V(Value)三个矩阵
 * 2. 计算注意力分数: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
 * 3. 使用多个头并行计算，增强模型表达能力
 */

import { 
  matrixMultiply, 
  transpose, 
  batchSoftmax, 
  scaleScores,
  generateCausalMask,
  applyMask,
  initializeWeights
} from '../utils/math.js';

/**
 * 单个注意力头
 */
class AttentionHead {
  /**
   * 构造函数
   * @param {number} dModel - 模型维度
   * @param {number} dK - Key/Query维度（通常为 dModel / numHeads）
   * @param {number} dV - Value维度（通常等于dK）
   */
  constructor(dModel, dK, dV) {
    this.dModel = dModel;
    this.dK = dK;
    this.dV = dV;
    
    // 初始化权重矩阵：W_q, W_k, W_v
    // 注意：每个头只输出d_v维度，不在这里做最终投影
    this.W_q = initializeWeights(dModel, dK); // Query权重
    this.W_k = initializeWeights(dModel, dK); // Key权重
    this.W_v = initializeWeights(dModel, dV); // Value权重
  }

  /**
   * 计算注意力
   * @param {number[][]} x - 输入矩阵 (seq_len x d_model)
   * @param {boolean} useCausalMask - 是否使用因果掩码（GPT-2需要）
   * @returns {number[][]} 注意力输出 (seq_len x d_v)
   */
  forward(x, useCausalMask = true) {
    const seqLen = x.length;
    
    // 1. 计算Q, K, V
    const Q = matrixMultiply(x, this.W_q); // (seq_len x d_k)
    const K = matrixMultiply(x, this.W_k); // (seq_len x d_k)
    const V = matrixMultiply(x, this.W_v); // (seq_len x d_v)
    
    // 2. 计算注意力分数: QK^T
    const K_transposed = transpose(K); // (d_k x seq_len)
    const scores = matrixMultiply(Q, K_transposed); // (seq_len x seq_len)
    
    // 3. 缩放: 除以 sqrt(d_k) 防止梯度消失
    const scale = 1 / Math.sqrt(this.dK);
    const scaledScores = scaleScores(scores, scale);
    
    // 4. 应用因果掩码（如果是GPT-2的自回归模式）
    let maskedScores = scaledScores;
    if (useCausalMask) {
      const mask = generateCausalMask(seqLen);
      maskedScores = applyMask(scaledScores, mask);
    }
    
    // 5. Softmax归一化得到注意力权重
    const attentionWeights = batchSoftmax(maskedScores); // (seq_len x seq_len)
    
    // 6. 加权求和: Attention * V
    // 每个头输出 (seq_len x d_v)，不在这里做最终投影
    const attentionOutput = matrixMultiply(attentionWeights, V); // (seq_len x d_v)
    
    return attentionOutput;
  }
}

/**
 * Multi-Head Attention（多头注意力）
 * 将输入分成多个头，每个头独立计算注意力，最后拼接
 */
export class MultiHeadAttention {
  /**
   * 构造函数
   * @param {number} dModel - 模型维度（例如512）
   * @param {number} numHeads - 注意力头数（例如8）
   */
  constructor(dModel, numHeads) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    
    // 确保dModel能被numHeads整除
    if (dModel % numHeads !== 0) {
      throw new Error(`dModel (${dModel}) 必须能被 numHeads (${numHeads}) 整除`);
    }
    
    // 每个头的维度
    this.dK = dModel / numHeads;
    this.dV = dModel / numHeads;
    
    // 创建多个注意力头
    this.heads = [];
    for (let i = 0; i < numHeads; i++) {
      this.heads.push(new AttentionHead(dModel, this.dK, this.dV));
    }
    
    // 最终的输出投影层（可选，用于进一步变换）
    this.W_o = initializeWeights(dModel, dModel);
  }

  /**
   * 前向传播
   * @param {number[][]} x - 输入矩阵 (seq_len x d_model)
   * @param {boolean} useCausalMask - 是否使用因果掩码
   * @returns {number[][]} 多头注意力输出 (seq_len x d_model)
   */
  forward(x, useCausalMask = true) {
    const seqLen = x.length;
    
    // 1. 每个头独立计算注意力
    const headOutputs = this.heads.map(head => 
      head.forward(x, useCausalMask)
    );
    
    // 2. 拼接所有头的输出
    // 每个头的输出是 (seq_len x d_model)，拼接后是 (seq_len x d_model)
    // 实际上每个头输出 (seq_len x d_v)，拼接后是 (seq_len x d_model)
    const concatenated = this.concatenateHeads(headOutputs, seqLen);
    
    // 3. 最终输出投影（可选）
    const output = matrixMultiply(concatenated, this.W_o);
    
    return output;
  }

  /**
   * 拼接多个头的输出
   * @param {number[][][]} headOutputs - 每个头的输出数组，每个头输出 (seq_len x d_v)
   * @param {number} seqLen - 序列长度
   * @returns {number[][]} 拼接后的矩阵 (seq_len x d_model)
   */
  concatenateHeads(headOutputs, seqLen) {
    // 每个头的输出是 (seq_len x d_v)
    // 拼接后应该是 (seq_len x d_model)，其中 d_model = numHeads * d_v
    const result = [];
    
    for (let s = 0; s < seqLen; s++) {
      const concatenated = [];
      // 对每个时间步，拼接所有头的特征
      for (let h = 0; h < this.numHeads; h++) {
        // headOutputs[h][s] 是第h个头在第s个时间步的d_v维向量
        concatenated.push(...headOutputs[h][s]);
      }
      result.push(concatenated);
    }
    
    return result;
  }

  /**
   * 设置权重（用于加载预训练模型）
   * @param {Object} weights - 权重对象
   */
  setWeights(weights) {
    // 这里可以设置预训练权重
    // 实际使用时需要根据权重格式进行适配
  }
}

