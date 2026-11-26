# awesome-Loop-Transformer [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[中文导读](README.zh-CN.md) · [Papers by topic](docs/research-and-code.md) · [Model weights](docs/models-and-weights.md) · [Reports and tools](docs/reports-and-tools.md) · [Other loops in language models](docs/language-model-loops.md)

A reading guide to Transformers that reuse layers across depth. Start with the architecture, follow its origins in adaptive computation and machine translation, then try varying the depth of a released language model.

## What is a Loop Transformer?

A **Loop Transformer** (also called a *looped* or *recurrent-depth Transformer*) applies the same Transformer block, or stack of blocks, repeatedly to an evolving hidden state. Each pass receives the state produced by the previous pass. In a language model, several such updates can happen while the token prefix stays fixed, before the model predicts the next token. [Universal Transformers](https://arxiv.org/abs/1807.03819) · [Huginn](https://arxiv.org/abs/2502.05171)

Imagine a recurrent core containing two layers, A and B. Running three passes executes **A → B → A → B → A → B**. That is six layer applications with two sets of layer weights. On its second visit, A receives the representation that B just updated. The model may also have input and output layers that run only once. [Recurrently stacked translation models](https://arxiv.org/abs/1807.05353) · [Huginn model card](https://huggingface.co/tomg-group-umd/huginn-0125)

**Who chooses the depth?** An architecture can prescribe a fixed number of passes, a caller can select a budget for a trained checkpoint, or the model can learn when each position should exit. These choices answer different questions. A six-pass model can save parameters at fixed depth. Changing the pass count of one checkpoint tests what extra inference compute buys. [ALBERT](https://arxiv.org/abs/1909.11942) · [Huginn](https://arxiv.org/abs/2502.05171) · [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)

## Where the idea came from

| First public release | What changed |
| --- | --- |
| 2016–17 · Two ingredients | Graves’s Adaptive Computation Time let an RNN learn how many internal updates to spend on an input. The original Transformer processed token positions in parallel using a stack of attention and feed-forward layers. The Universal Transformer later brought adaptive halting into that architecture. [ACT](https://arxiv.org/abs/1603.08983) · [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Universal Transformer, §2](https://arxiv.org/html/1807.03819v3) |
| July 2018 · Two motivations | Dehghani et al. studied whether repeatedly updating every position helps with tasks such as copying and logical inference. The same month, Dabre and Fujita asked how much translation quality survives weight sharing. On their translation experiments, one layer used six times gave quality comparable to six separate layers. [Universal Transformers](https://arxiv.org/abs/1807.03819) · [Recurrent Stacking of Layers for Compact Neural Machine Translation Models](https://arxiv.org/abs/1807.05353) |
| 2019 · Sharing during pretraining | ALBERT used cross-layer sharing in a BERT-style encoder. Its ablations are worth reading because the model also changes the embedding parameterization. [ALBERT](https://arxiv.org/abs/1909.11942) |
| 2023–25 · Learning an update and spending more steps | Yang et al. studied what a shared update learns on in-context data-fitting tasks. Geiping et al. later pretrained Huginn, a 3.5B language model whose recurrent core can run for different numbers of steps at inference. [In-context learning study](https://arxiv.org/abs/2311.12424) · [Huginn report](https://arxiv.org/abs/2502.05171) |

## Five papers to read in order

1. **Understand one pass.** Start with Figure 2 of *Universal Transformers*. Pick a token and follow its state through two steps. Attention reads the other positions, the shared transformation updates the state, and the adaptive variant decides which positions continue. [Paper and figure](https://arxiv.org/html/1807.03819v3)
2. **See what sharing can help a model learn.** *Looped Transformers are Better at Learning Learning Algorithms* compares looped and untied models on in-context data fitting. Start with linear regression, where the target and the prediction error are explicit. Its parameter-efficiency result applies to the tasks tested. [Paper](https://arxiv.org/abs/2311.12424) · [Experiments](https://github.com/Leiay/looped_transformer)
3. **Change depth yourself.** In *Scaling up Test-Time Compute with Latent Reasoning*, find the evaluation that varies recurrence on the **same** Huginn checkpoint. Its model card exposes `num_steps`. Try 8 and 32 on the same arithmetic prompts with greedy decoding, then compare answer accuracy and elapsed time. [Paper](https://arxiv.org/abs/2502.05171) · [Checkpoint and usage](https://huggingface.co/tomg-group-umd/huginn-0125)
4. **Ask what happens between passes.** *Latent Chain-of-Thought?* probes Huginn on arithmetic. Compare what the Logit Lens and Coda Lens recover at the same depth. The evidence for readable intermediate arithmetic steps depends on the probe and the layer. [Paper](https://arxiv.org/abs/2507.02199) · [Probe code](https://github.com/wenquanlu/huginn-latent-cot)
5. **Put a price on a recurrence.** *How Much Is One Recurrence Worth?* compares runs at matched executed depth and measures validation loss. In one example, a 410M model running four passes matches the loss of a 580M untied model at roughly the training cost of a 1B untied model. Track unique parameters, layer applications and training compute separately. [Paper](https://arxiv.org/abs/2604.21106) · [Scaling experiments](https://github.com/kschwethelm/looped-lm-scaling)

## Explore the collection

| If you want to… | Go to |
| --- | --- |
| Follow a research question across papers | [Papers by topic](docs/research-and-code.md), including [algorithm learning](docs/research-and-code.md#algorithms-and-length-generalization), [adaptive depth](docs/research-and-code.md#depth-allocation-and-scaling) and [inference systems](docs/research-and-code.md#inference-attention-and-memory) |
| Download a model or compare released variants | [Models, weight files and licenses](docs/models-and-weights.md) |
| Find a training recipe, evaluation harness or hidden-state probe | [Technical reports, implementations and analysis tools](docs/reports-and-tools.md) |
| Compare loops elsewhere in a language model | [Other language model loops](docs/language-model-loops.md) covers segment memory in Transformer-XL, latent thoughts in Coconut, denoising in LLaDA and feedback in Self-Refine. |

[Suggest a paper, model or report](CONTRIBUTING.md) · [Related reading lists](docs/reports-and-tools.md#related-lists)
