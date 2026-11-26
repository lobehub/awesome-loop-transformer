# Awesome Looped Transformers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A Looped Transformer applies the same Transformer block or layer stack more than once across effective depth. The parameters can stay fixed while the amount of computation changes. This list connects architectural papers to model releases, downloadable weights, and experiments on when another pass helps.

## Contents

- [Start here](#start-here)
- [Models and weights](#models-and-weights)
- [Research papers](#research-papers)
- [Technical reports and implementations](#technical-reports-and-implementations)
- [Related lists](#related-lists)

## Start here

- **Understand the architecture.** [Universal Transformers](https://arxiv.org/abs/1807.03819) shows how a shared update can refine every token’s representation across depth. Its adaptive variant lets individual positions stop at different steps.
- **Try changing depth on one checkpoint.** [Huginn’s model card](https://huggingface.co/tomg-group-umd/huginn-0125) gives inference examples and identifies the recurrent core, input layers, and output layers. Vary the recurrence count while keeping the prompt and decoding settings fixed.
- **Interpret a scaling claim.** [How Much Is One Recurrence Worth?](https://arxiv.org/abs/2604.21106) compares tied and untied models at the same executed depth. In one reported configuration, a 410M-parameter model with four recurrences matched the validation loss of a 580M untied model at roughly the training cost of a 1B untied model. Parameter savings and compute savings answer different questions.

## Models and weights

Checkpoint links open the publisher’s file tree, including all shards when a checkpoint is split. The 12 releases below had publicly listed SafeTensors files on 2026-11-26. The model cards declare Apache-2.0 for these checkpoints.

| Model family | What it offers | Official weight files | Paper or report |
| --- | --- | --- | --- |
| ALBERT | A fixed-depth encoder with cross-layer parameter sharing. The base-v2 release is a compact reference for studying weight tying in masked language modeling. | [Base v2](https://huggingface.co/albert/albert-base-v2/tree/main) | [ALBERT](https://arxiv.org/abs/1909.11942) |
| Huginn | A 3.5B language model with a recurrent core between distinct input and output stacks. The same checkpoint supports different recurrence counts at inference. | [Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125/tree/main) | [Recurrent depth report](https://arxiv.org/abs/2502.05171) |
| Ouro | Shared-weight language models with a learned exit mechanism. Base and Thinking releases are available at both sizes. | [1.4B base](https://huggingface.co/ByteDance/Ouro-1.4B/tree/main) · [1.4B Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/tree/main) · [2.6B base](https://huggingface.co/ByteDance/Ouro-2.6B/tree/main) · [2.6B Thinking](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/tree/main) | [Ouro report](https://arxiv.org/abs/2510.25741) |
| LoopCoder-v2 | A 7B instruction-tuned coding model. The downloadable variant runs its 14 shared layers twice. | [7B two-loop checkpoint](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2/tree/main) | [Loop-count study](https://arxiv.org/abs/2606.18023) |
| Nanbeige4.2 | A shared layer stack is reused during a forward pass. The report covers the base model and post-training for tool use and coding. The reported 3B count excludes embeddings. | [3B base](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base/tree/main) · [3B post-trained](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/main) | [Nanbeige4.2 technical report](https://arxiv.org/abs/2607.22083) |
| ARLoop | Uses attention over earlier layer states to decide what each layer reads. The authors also release a 0.6B looped baseline with standard pre-normalization for a same-size residual-design comparison. | [0.6B](https://huggingface.co/osunlp/ARLoop-0.6B/tree/main) · [0.6B baseline](https://huggingface.co/osunlp/ARLoop-baseline-0.6B/tree/main) · [1.2B](https://huggingface.co/osunlp/ARLoop-1.2B/tree/main) | [ARLoop report](https://arxiv.org/abs/2611.05234) |

## Research papers

### Sharing weights and learning algorithms

- [Universal Transformers](https://arxiv.org/abs/1807.03819) — Dehghani et al. introduce recurrence across depth and evaluate both fixed and adaptive computation on sequence tasks.
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) — Lan et al. study cross-layer parameter sharing alongside factorized embeddings. The ablations help separate the two sources of parameter savings. [Implementation](https://github.com/google-research/albert)
- [Looped Transformers are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424) — On in-context data-fitting tasks, the authors report that looped models can match standard Transformers with less than a tenth of the parameters. The experiments concern learning algorithms on synthetic problems. [Experiments](https://github.com/Leiay/looped_transformer)

### Allocating depth and testing its value

- [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524) — A router assigns each token a recurrence depth. The equal-training-FLOP comparisons examine whether concentrating computation on active tokens improves perplexity and downstream accuracy. [Code](https://github.com/raymin0223/mixture_of_recursions)
- [How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models](https://arxiv.org/abs/2604.21106) — Sweeps one, two, four, and eight recurrences to estimate how much extra unique parameter capacity a shared pass can replace at a given validation loss. [Scaling experiments](https://github.com/kschwethelm/looped-lm-scaling)
- [LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling](https://arxiv.org/abs/2606.18023) — Compares separately trained coding models with different loop counts. Two passes improve on the untied baseline in the authors’ setup, while three or more regress. This tests a training design choice; changing one released checkpoint’s inference depth is a different experiment.

### Training across loop depths

- [LoopRPT: Reinforcement Pre-Training for Looped Language Models](https://openreview.net/forum?id=h8EZBjmDcp) — Rewards intermediate steps for progress on next-token prediction relative to a moving-average teacher. Figure 9 compares moving, frozen, and absent teacher references.
- [Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models](https://arxiv.org/abs/2610.10623) — On student-generated trajectories, the terminal loop teaches an earlier loop. The authors compare freezing the teacher with refreshing it through shared-weight updates. The [project page](https://wanglne.github.io/LoopOPD-page/) tracks the code and model release.

## Technical reports and implementations

| Release | What to look for in the report | Author resources |
| --- | --- | --- |
| [Huginn recurrent-depth report](https://arxiv.org/abs/2502.05171) | How the recurrent block is pretrained and how performance changes as more compute is spent per token at inference. | [Training and evaluation code](https://github.com/seal-rg/recurrent-pretraining) · [Training dataset](https://huggingface.co/datasets/tomg-group-umd/huginn-dataset) |
| [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) | Ouro’s pretraining recipe and the objective used to learn when to exit. Its model card explains the runtime settings that enable adaptive exit. | [Project](https://ouro-llm.github.io/) · [Model card and exit settings](https://huggingface.co/ByteDance/Ouro-1.4B) |
| [Nanbeige4.2-3B technical report](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/Nanbeige42_report.pdf) | How a looped backbone fits into a complete 28T-token pretraining and agent post-training pipeline. The final agent benchmarks reflect that whole pipeline. | [Architecture implementation](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/modeling_nanbeige.py) |
| [LoopCoder-v2 report](https://arxiv.org/abs/2606.18023) | Why cross-loop position offsets and cache sharing matter for parallel looping, and how the authors tested the cost of another pass. | [Implementation](https://github.com/CSJianYang/LoopCoder) · [Released configuration](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2/blob/main/config.json) |
| [ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals](https://arxiv.org/abs/2611.05234) | Sliding-window versus full-history attention across depth, followed by larger pretraining runs. The released baseline is useful when comparing residual designs. | [Pretraining, inference, and evaluation](https://github.com/OSU-NLP-Group/ARLoop) |

## Related lists

- [Awesome Loop Models](https://github.com/huskydoge/Awesome-Loop-Models) — A broader catalog covering looped architectures, implicit models, theory, and applications, with an interactive paper browser.
- [HJSang’s Awesome Looped Transformers](https://github.com/HJSang/awesome-looped-transformers) — A complementary reading list covering recurrent-depth Transformers, systems work, and commentary.

See the [contribution guide](CONTRIBUTING.md) to suggest a paper, checkpoint, or report.
