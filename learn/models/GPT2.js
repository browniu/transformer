export class GPT2 {
  constructor(config) {
    this.vocabSize = config.vocabSize;
    this.dModel = config.d.Model;
  }

  getParameterCount() {
    return 0;
  }
}