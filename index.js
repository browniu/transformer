/**
 * GPT-2 Transformer 模型入口文件
 * 导出所有核心模块和模型类
 */

// 导出核心组件
export { LayerNorm } from './src/core/LayerNorm.js';
export { MultiHeadAttention } from './src/core/Attention.js';
export { FeedForward } from './src/core/FeedForward.js';
export { TokenEmbedding } from './src/core/Embedding.js';
export { LearnedPositionalEncoding, SinusoidalPositionalEncoding } from './src/core/PositionalEncoding.js';

// 导出模型
export { TransformerBlock } from './src/models/TransformerBlock.js';
export { GPT2 } from './src/models/GPT2.js';

// 导出工具函数
export * from './src/utils/math.js';

