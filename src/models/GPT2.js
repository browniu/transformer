/**
 * GPT-2模型主类
 * 完整的GPT-2架构实现
 * 
 * 架构组成：
 * 1. Token Embedding（词嵌入）
 * 2. Positional Encoding（位置编码）
 * 3. N个Transformer Blocks（Transformer块）
 * 4. Layer Normalization（最终层归一化）
 * 5. Language Model Head（语言模型头，用于预测下一个token）
 * 
 * GPT-2特点：
 * - 仅使用解码器（Decoder-only）
 * - 自回归生成（Autoregressive）
 * - 使用因果掩码（Causal Mask）
 */

import { TokenEmbedding } from '../core/Embedding.js';
import { LearnedPositionalEncoding } from '../core/PositionalEncoding.js';
import { TransformerBlock } from './TransformerBlock.js';
import { LayerNorm } from '../core/LayerNorm.js';
import { matrixAdd, matrixMultiply, batchSoftmax, initializeWeights } from '../utils/math.js';

/**
 * GPT-2模型类
 */
export class GPT2 {
  /**
   * 构造函数
   * @param {Object} config - 模型配置
   * @param {number} config.vocabSize - 词汇表大小（例如50257）
   * @param {number} config.dModel - 模型维度（例如768）
   * @param {number} config.numLayers - Transformer层数（例如12）
   * @param {number} config.numHeads - 注意力头数（例如12）
   * @param {number} config.dFF - 前馈网络隐藏层维度（通常是dModel的4倍）
   * @param {number} config.maxSeqLen - 最大序列长度（例如1024）
   * @param {number} config.dropout - Dropout率（可选）
   */
  constructor(config) {
    // 模型配置
    this.vocabSize = config.vocabSize;
    this.dModel = config.dModel;
    this.numLayers = config.numLayers;
    this.numHeads = config.numHeads;
    this.dFF = config.dFF || config.dModel * 4; // 默认4倍
    this.maxSeqLen = config.maxSeqLen;
    this.dropout = config.dropout || 0.0;
    
    // 验证配置
    if (this.dModel % this.numHeads !== 0) {
      throw new Error(`dModel (${this.dModel}) 必须能被 numHeads (${this.numHeads}) 整除`);
    }
    
    // === 1. Token Embedding层 ===
    this.tokenEmbedding = new TokenEmbedding(this.vocabSize, this.dModel);
    
    // === 2. Positional Encoding层 ===
    this.positionalEncoding = new LearnedPositionalEncoding(
      this.maxSeqLen, 
      this.dModel
    );
    
    // === 3. Transformer Blocks ===
    this.transformerBlocks = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.transformerBlocks.push(
        new TransformerBlock(
          this.dModel,
          this.numHeads,
          this.dFF,
          this.dropout
        )
      );
    }
    
    // === 4. 最终Layer Normalization ===
    this.finalLayerNorm = new LayerNorm(this.dModel);
    
    // === 5. Language Model Head（输出层）===
    // 将d_model维度的隐藏状态映射到vocab_size维度的logits
    this.lmHead = initializeWeights(this.dModel, this.vocabSize);
    this.lmBias = Array(this.vocabSize).fill(0);
  }

  /**
   * 前向传播
   * @param {number[]} tokenIds - Token ID数组 (seq_len)
   * @param {boolean} returnLogits - 是否返回logits（用于训练），否则返回概率
   * @returns {Object} 包含logits和hiddenStates的对象
   */
  forward(tokenIds, returnLogits = true) {
    const seqLen = tokenIds.length;
    
    if (seqLen > this.maxSeqLen) {
      throw new Error(`序列长度 ${seqLen} 超过最大长度 ${this.maxSeqLen}`);
    }
    
    // === 1. Token Embedding ===
    // 将token ID转换为嵌入向量
    const tokenEmbeddings = this.tokenEmbedding.forward(tokenIds);
    // tokenEmbeddings: (seq_len x d_model)
    
    // === 2. Positional Encoding ===
    // 添加位置信息
    const positionEmbeddings = this.positionalEncoding.forward(seqLen);
    // positionEmbeddings: (seq_len x d_model)
    
    // === 3. 合并Token和Position Embeddings ===
    let hiddenStates = matrixAdd(tokenEmbeddings, positionEmbeddings);
    // hiddenStates: (seq_len x d_model)
    
    // === 4. 通过所有Transformer Blocks ===
    for (let i = 0; i < this.numLayers; i++) {
      hiddenStates = this.transformerBlocks[i].forward(
        hiddenStates, 
        true // 使用因果掩码
      );
    }
    
    // === 5. 最终Layer Normalization ===
    const normalized = this.finalLayerNorm.forward(hiddenStates);
    
    // === 6. Language Model Head ===
    // 计算logits: (seq_len x vocab_size)
    const logits = matrixMultiply(normalized, this.lmHead);
    const logitsWithBias = logits.map(row => 
      row.map((val, i) => val + this.lmBias[i])
    );
    
    // 如果不需要logits，返回概率分布
    if (!returnLogits) {
      const probs = batchSoftmax(logitsWithBias);
      return {
        logits: logitsWithBias,
        probabilities: probs,
        hiddenStates: normalized
      };
    }
    
    return {
      logits: logitsWithBias,
      hiddenStates: normalized
    };
  }

  /**
   * 生成下一个token（自回归生成）
   * @param {number[]} tokenIds - 当前序列的token IDs
   * @param {number} temperature - 温度参数（控制随机性，1.0为标准）
   * @param {number} topK - Top-K采样（可选，只从概率最高的K个token中选择）
   * @returns {number} 下一个token的ID
   */
  generateNextToken(tokenIds, temperature = 1.0, topK = null) {
    // 前向传播获取logits
    const { logits } = this.forward(tokenIds, true);
    
    // 取最后一个时间步的logits（预测下一个token）
    const lastLogits = logits[logits.length - 1];
    
    // 应用温度缩放
    const scaledLogits = lastLogits.map(val => val / temperature);
    
    // Top-K采样（如果指定）
    let candidateLogits = scaledLogits;
    if (topK !== null && topK < this.vocabSize) {
      // 找到top-K的索引
      const indexed = scaledLogits.map((val, idx) => ({ val, idx }));
      indexed.sort((a, b) => b.val - a.val);
      const topKIndices = new Set(indexed.slice(0, topK).map(item => item.idx));
      
      // 将非top-K的logits设为负无穷
      candidateLogits = scaledLogits.map((val, idx) => 
        topKIndices.has(idx) ? val : -Infinity
      );
    }
    
    // Softmax得到概率分布
    const probs = batchSoftmax([candidateLogits])[0];
    
    // 根据概率分布采样
    const random = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (random <= cumulative) {
        return i;
      }
    }
    
    // 如果由于浮点误差没有返回，返回最后一个
    return probs.length - 1;
  }

  /**
   * 生成完整序列（自回归生成）
   * @param {number[]} initialTokens - 初始token序列
   * @param {number} maxLength - 最大生成长度
   * @param {number} temperature - 温度参数
   * @param {number} topK - Top-K采样
   * @returns {number[]} 生成的完整token序列
   */
  generate(initialTokens, maxLength, temperature = 1.0, topK = null) {
    let tokens = [...initialTokens];
    
    for (let i = 0; i < maxLength; i++) {
      // 如果序列太长，只使用最后maxSeqLen个token
      const inputTokens = tokens.length > this.maxSeqLen 
        ? tokens.slice(-this.maxSeqLen)
        : tokens;
      
      // 生成下一个token
      const nextToken = this.generateNextToken(inputTokens, temperature, topK);
      tokens.push(nextToken);
      
      // 可以在这里添加停止条件（例如遇到结束token）
      // if (nextToken === END_TOKEN_ID) break;
    }
    
    return tokens;
  }

  /**
   * 获取模型参数数量（用于统计）
   * @returns {number} 参数总数
   */
  getParameterCount() {
    // 这是一个简化的估算，实际应该遍历所有参数
    let count = 0;
    
    // Embedding参数
    count += this.vocabSize * this.dModel; // Token embedding
    count += this.maxSeqLen * this.dModel; // Position embedding
    
    // Transformer Blocks参数
    // 每个block包含：
    // - Attention: 4个权重矩阵 (Q, K, V, O)
    // - Feed Forward: 2个权重矩阵和偏置
    // - Layer Norm: gamma和beta
    const paramsPerBlock = 
      this.numHeads * (4 * this.dModel * (this.dModel / this.numHeads)) + // Attention
      2 * this.dModel * this.dFF + this.dFF + this.dModel + // Feed Forward
      2 * this.dModel; // Layer Norm
    
    count += this.numLayers * paramsPerBlock;
    
    // Final Layer Norm
    count += 2 * this.dModel;
    
    // Language Model Head
    count += this.dModel * this.vocabSize + this.vocabSize;
    
    return count;
  }

  /**
   * 设置权重（用于加载预训练模型）
   * @param {Object} weights - 权重对象
   */
  setWeights(weights) {
    if (weights.tokenEmbedding) {
      this.tokenEmbedding.setEmbedding(weights.tokenEmbedding);
    }
    if (weights.positionalEncoding) {
      this.positionalEncoding.setPositionEmbedding(weights.positionalEncoding);
    }
    if (weights.transformerBlocks) {
      weights.transformerBlocks.forEach((blockWeights, i) => {
        this.transformerBlocks[i].setWeights(blockWeights);
      });
    }
    if (weights.finalLayerNorm) {
      this.finalLayerNorm.setParams(
        weights.finalLayerNorm.gamma,
        weights.finalLayerNorm.beta
      );
    }
    if (weights.lmHead) {
      this.lmHead = weights.lmHead;
    }
    if (weights.lmBias) {
      this.lmBias = weights.lmBias;
    }
  }
}

