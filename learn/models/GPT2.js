import {TokenEmbedding} from "../../src/core/Embedding.js";
import {LearnedPositionalEncoding} from "../../src/core/PositionalEncoding.js";
import {TransformerBlock} from "../../src/models/TransformerBlock.js";
import {LayerNorm} from "../../src/core/LayerNorm.js";
import {matrixAdd, matrixMultiply, batchSoftmax, initializeWeights} from "../../src/utils/math.js";

export class GPT2 {
    constructor(config) {
        this.vocabSize = config.vocabSize;
        this.dModel = config.dModel;
        this.numLayers = config.numLayers;
        this.numHeads = config.numHeads
        this.dFF = config.dFF || config.dModel * 4;
        this.maxSeqLen = config.maxSeqLen;
        this.dropout = config.dropout || 0.0;

        if (this.dModel % this.numHeads !== 0) {
            throw new Error(`dModel（${this.dModel}）必须能够被numHeads（${this.numHeads}）整除`);
        }

        this.tokenEmbedding = new TokenEmbedding(this.vocabSize, this.dModel);

        this.positionalEncoding = new LearnedPositionalEncoding(this.maxSeqLen, this.dModel);

        this.transformerBlocks = [];
        for (let i = 0; i < this.numLayers; i++) {
            this.transformerBlocks.push(new TransformerBlock(this.dModel, this.numHeads, this.dFF, this.dropout))
        }

        this.finalLayerNorm = new LayerNorm(this.dModel);

        this.lmHead = initializeWeights(this.dModel, this.vocabSize)
        this.lmBias = Array(this.vocabSize).fill(0);
    }

    forward(tokenIds, returnLogits = true) {
        const seqLen = tokenIds.length;

        if (seqLen > this.maxSeqLen) {
            throw new Error(`序列长度（${seqLen}）超出最大长度（${this.maxSeqLen}）`)
        }

        const tokenEmbeddings = this.tokenEmbedding.forward(tokenIds)

        const positionEmbeddings = this.positionalEncoding.forward(seqLen)

        let hiddenStates = matrixAdd(tokenEmbeddings, positionEmbeddings)

        for (let i = 0; i < this.numLayers; i++) {
            hiddenStates = this.transformerBlocks[i].forward(hiddenStates, true)
        }

        const normalized = this.finalLayerNorm.forward(hiddenStates)

        const logits = matrixMultiply(normalized, this.lmHead)
        const logitsWithBias = logits.map(row => {
            return row.map((val, i) => val + this.lmBias[i])
        })

        if (!returnLogits) {
            const probabilities = batchSoftmax(logitsWithBias)
            return {
                logits: logitsWithBias,
                probabilities: probabilities,
                hiddenStates: normalized
            }
        }

        return {
            logits: logitsWithBias,
            hiddenStates: normalized
        }

    }


    getParameterCount() {
        return 0;
    }
}
