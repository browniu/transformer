/**
 * GPT-2 Transformer 使用示例
 * 演示如何创建模型、进行前向传播和文本生成
 */

import { GPT2 } from './index.js';

// ========== 配置参数 ==========
// 这里使用GPT-2 Small的配置（简化版用于演示）
const config = {
  vocabSize: 50257,      // GPT-2的词汇表大小
  dModel: 768,           // 模型维度
  numLayers: 12,         // Transformer层数
  numHeads: 12,          // 注意力头数
  dFF: 3072,             // 前馈网络隐藏层维度 (4 * dModel)
  maxSeqLen: 1024,       // 最大序列长度
  dropout: 0.1           // Dropout率（本实现暂未使用）
};

// ========== 创建模型 ==========
console.log('正在创建GPT-2模型...');
const model = new GPT2(config);

// 打印模型信息
console.log(`模型创建成功！`);
console.log(`参数数量: ${model.getParameterCount().toLocaleString()}`);
console.log(`模型维度: ${model.dModel}`);
console.log(`层数: ${model.numLayers}`);
console.log(`注意力头数: ${model.numHeads}`);
console.log('');

// ========== 示例1: 前向传播 ==========
console.log('=== 示例1: 前向传播 ===');
// 假设的token IDs（实际使用时需要tokenizer将文本转换为ID）
const tokenIds = [15496, 11, 995, 318, 13779, 13]; // "Hello, how are you?"

console.log(`输入token IDs: [${tokenIds.join(', ')}]`);
console.log(`序列长度: ${tokenIds.length}`);

// 前向传播
const output = model.forward(tokenIds, true);

console.log(`输出logits形状: (${output.logits.length} x ${output.logits[0].length})`);
console.log(`隐藏状态形状: (${output.hiddenStates.length} x ${output.hiddenStates[0].length})`);

// 显示最后一个时间步的top-5预测
const lastLogits = output.logits[output.logits.length - 1];
const top5Indices = lastLogits
  .map((val, idx) => ({ val, idx }))
  .sort((a, b) => b.val - a.val)
  .slice(0, 5);

console.log('\nTop-5 预测token IDs:');
top5Indices.forEach((item, i) => {
  console.log(`  ${i + 1}. Token ID: ${item.idx}, Logit: ${item.val.toFixed(4)}`);
});
console.log('');

// ========== 示例2: 生成下一个token ==========
console.log('=== 示例2: 生成下一个token ===');
const prompt = [15496, 11]; // "Hello,"

console.log(`输入prompt: [${prompt.join(', ')}]`);

// 生成下一个token（使用不同的温度参数）
const temperatures = [0.5, 1.0, 1.5];
temperatures.forEach(temp => {
  const nextToken = model.generateNextToken(prompt, temp);
  console.log(`  温度 ${temp}: 下一个token ID = ${nextToken}`);
});
console.log('');

// ========== 示例3: 自回归生成 ==========
console.log('=== 示例3: 自回归生成序列 ===');
const initialTokens = [15496]; // "Hello"

console.log(`初始token: [${initialTokens.join(', ')}]`);
console.log('开始生成...');

// 生成10个新token
const generated = model.generate(initialTokens, 10, 1.0, null);

console.log(`生成的完整序列 (${generated.length}个tokens):`);
console.log(`  [${generated.join(', ')}]`);
console.log('');

// ========== 示例4: 使用Top-K采样 ==========
console.log('=== 示例4: 使用Top-K采样生成 ===');
const topKPrompt = [15496, 11]; // "Hello,"

console.log(`输入prompt: [${topKPrompt.join(', ')}]`);

// 使用Top-K=10采样
const nextTokenTopK = model.generateNextToken(topKPrompt, 1.0, 10);
console.log(`Top-K=10采样结果: token ID = ${nextTokenTopK}`);

// 生成序列（Top-K=5）
const generatedTopK = model.generate(initialTokens, 10, 1.0, 5);
console.log(`Top-K=5生成的序列: [${generatedTopK.join(', ')}]`);
console.log('');

// ========== 模型架构说明 ==========
console.log('=== 模型架构说明 ===');
console.log('GPT-2 Transformer架构包含以下组件:');
console.log('1. Token Embedding: 将token ID转换为向量');
console.log('2. Positional Encoding: 添加位置信息（可学习的）');
console.log('3. Transformer Blocks (×12):');
console.log('   - Multi-Head Self-Attention (12 heads)');
console.log('   - Feed Forward Network (GELU激活)');
console.log('   - Layer Normalization (Pre-LN)');
console.log('   - Residual Connections');
console.log('4. Final Layer Normalization');
console.log('5. Language Model Head: 输出词汇表大小的logits');
console.log('');
console.log('特点:');
console.log('- 仅使用解码器（Decoder-only）');
console.log('- 自回归生成（Autoregressive）');
console.log('- 使用因果掩码（Causal Mask）防止看到未来token');
console.log('- Pre-LN架构（Layer Norm在子层之前）');

