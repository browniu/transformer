import {GPT2} from './models/GPT2.js';

const config = {
    vocabSize: 100, // GPT-2的词汇表大小
    dModel: 3, // 模型维度
    numLayers: 3, // Transformer层数
    numHeads: 3, // 注意力头数
    dFF: 4, // 前馈网络隐藏层维度 (4 * dModel)
    maxSeqLen: 10, // 最大序列长度
    dropout: 0.1 // Dropout率（本实现暂未使用）
};

console.log('正在创建GPT-2模型...');
const model = new GPT2(config);

console.log('GPT-2模型创建成功！');

console.log('模型参数数量: ', model.getParameterCount());
console.log('模型维度: ', model.dModel);
console.log('层数: ', model.numLayers);
console.log('注意力头数: ', model.numHeads);
console.log('前馈网络隐藏层维度: ', model.dFF);
console.log('最大序列长度: ', model.maxSeqLen);
console.log('Dropout率: ', model.dropout);

console.log('=== 示例1：向前传播 ===')
const tokenIds = [1, 32, 43, 64, 41, 15];

console.log(`输入token IDs: ${tokenIds.join(', ')}`);
console.log(`序列长度: ${tokenIds.length}`);

const output = model.forward(tokenIds, true);

console.log(`输出logits形状: (${output.logits.length} x ${output.logits[0].length})`);
console.log(`隐藏状态形状: (${output.hiddenStates.length} x ${output.hiddenStates[0].length})`);

const lastLogits = output.logits[output.logits.length - 1];
const top5Indices = lastLogits.map((val, idx) => ({val, idx})).sort((a, b) => b.val - a.val).slice(0, 5)

console.log('Top-5 预测Token IDs：', top5Indices)
