import { GPT2 } from './models/GPT2.js';

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

