/**
 * 数学工具函数模块
 * 提供矩阵运算、激活函数等基础数学操作
 */

/**
 * 矩阵乘法
 * @param {number[][]} a - 矩阵A (m x n)
 * @param {number[][]} b - 矩阵B (n x p)
 * @returns {number[][]} 结果矩阵 (m x p)
 */
export function matrixMultiply(a, b) {
  const m = a.length;
  const n = a[0].length;
  const p = b[0].length;
  const result = Array(m).fill(0).map(() => Array(p).fill(0));

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += a[i][k] * b[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

/**
 * 矩阵转置
 * @param {number[][]} matrix - 输入矩阵
 * @returns {number[][]} 转置后的矩阵
 */
export function transpose(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const result = Array(cols).fill(0).map(() => Array(rows).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = matrix[i][j];
    }
  }
  return result;
}

/**
 * Softmax函数 - 将向量转换为概率分布
 * @param {number[]} x - 输入向量
 * @returns {number[]} 概率分布向量
 */
export function softmax(x) {
  // 数值稳定性：减去最大值防止溢出
  const max = Math.max(...x);
  const exp = x.map(val => Math.exp(val - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(val => val / sum);
}

/**
 * 批量Softmax - 对矩阵的每一行应用softmax
 * @param {number[][]} matrix - 输入矩阵 (batch_size x seq_len)
 * @returns {number[][]} 每行都是概率分布的矩阵
 */
export function batchSoftmax(matrix) {
  return matrix.map(row => softmax(row));
}

/**
 * 缩放点积注意力分数
 * @param {number[][]} scores - 注意力分数矩阵
 * @param {number} scale - 缩放因子 (通常是 1/sqrt(d_k))
 * @returns {number[][]} 缩放后的注意力分数
 */
export function scaleScores(scores, scale) {
  return scores.map(row => row.map(val => val * scale));
}

/**
 * 生成因果掩码（Causal Mask）- 用于GPT-2的自回归特性
 * 防止模型看到未来的token
 * @param {number} seqLen - 序列长度
 * @returns {number[][]} 掩码矩阵，下三角为0，上三角为-Inf
 */
export function generateCausalMask(seqLen) {
  const mask = Array(seqLen).fill(0).map(() => Array(seqLen).fill(0));
  for (let i = 0; i < seqLen; i++) {
    for (let j = i + 1; j < seqLen; j++) {
      mask[i][j] = -Infinity; // 上三角设为负无穷，softmax后为0
    }
  }
  return mask;
}

/**
 * 应用掩码到注意力分数
 * @param {number[][]} scores - 注意力分数矩阵
 * @param {number[][]} mask - 掩码矩阵
 * @returns {number[][]} 应用掩码后的分数
 */
export function applyMask(scores, mask) {
  return scores.map((row, i) => 
    row.map((val, j) => val + mask[i][j])
  );
}

/**
 * 矩阵加法
 * @param {number[][]} a - 矩阵A
 * @param {number[][]} b - 矩阵B
 * @returns {number[][]} 结果矩阵
 */
export function matrixAdd(a, b) {
  return a.map((row, i) => 
    row.map((val, j) => val + b[i][j])
  );
}

/**
 * 矩阵元素级乘法
 * @param {number[][]} a - 矩阵A
 * @param {number[][]} b - 矩阵B
 * @returns {number[][]} 结果矩阵
 */
export function matrixElementWiseMultiply(a, b) {
  return a.map((row, i) => 
    row.map((val, j) => val * b[i][j])
  );
}

/**
 * GELU激活函数 - GPT-2使用的激活函数
 * @param {number} x - 输入值
 * @returns {number} 激活后的值
 */
export function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

/**
 * 批量GELU - 对矩阵的每个元素应用GELU
 * @param {number[][]} matrix - 输入矩阵
 * @returns {number[][]} 激活后的矩阵
 */
export function batchGelu(matrix) {
  return matrix.map(row => row.map(val => gelu(val)));
}

/**
 * 初始化权重矩阵 - Xavier初始化
 * @param {number} rows - 行数
 * @param {number} cols - 列数
 * @returns {number[][]} 初始化的权重矩阵
 */
export function initializeWeights(rows, cols) {
  const limit = Math.sqrt(6.0 / (rows + cols));
  return Array(rows).fill(0).map(() => 
    Array(cols).fill(0).map(() => 
      (Math.random() * 2 - 1) * limit
    )
  );
}

