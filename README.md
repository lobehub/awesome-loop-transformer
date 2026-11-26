# awesome-Loop-Transformer [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Papers, models, downloadable weights and technical reports on repeated computation in language models. A loop can refine a hidden state, carry memory into the next segment, revise a draft, act on feedback or produce the training data for a later model. The Looped Transformer collection also covers related work in vision, control and algorithmic reasoning.

## Contents

- [Find a loop by mechanism](#find-a-loop-by-mechanism)
  - [Language model papers and implementations](docs/language-model-loops.md)
- [Start here](#start-here)
- [Models and weights](#models-and-weights)
- [Looped Transformer papers](#research-papers)
  - [More papers and implementations](docs/research-and-code.md)
- [Technical reports and implementations](#technical-reports-and-implementations)
- [Analysis tools](#analysis-tools)
- [Related lists](#related-lists)

## Find a loop by mechanism

| Where the loop runs | What goes into the next step | Examples |
| --- | --- | --- |
| Shared depth or equilibrium | A hidden state revisits the same block or a solver refines it toward a fixed point. | [Looped Transformers](#research-papers) · [Equilibrium models](docs/language-model-loops.md#recurrent-depth-and-equilibrium) |
| Sequence memory | The next token or segment receives a recurrent state, cached activations or memory tokens. | [Transformer-XL, RWKV, Mamba](docs/language-model-loops.md#memory-across-tokens-and-segments) |
| Intermediate reasoning | The next step consumes a rationale token, a pause position or a continuous thought vector. | [Quiet-STaR, Coconut](docs/language-model-loops.md#intermediate-steps-before-an-answer) |
| Iterative decoding | The next pass edits a draft or fills part of a masked sequence. | [Mask-Predict, LLaDA](docs/language-model-loops.md#editing-and-denoising-text) |
| Feedback and search | A new model call receives a critique, observation, retrieved passage or partially explored branch. | [Self-Refine, ReAct, Tree of Thoughts](docs/language-model-loops.md#feedback-retrieval-and-search) |
| Training across rounds | Generated examples or a previous policy supply data for the next model update. | [STaR, ReST, SPIN](docs/language-model-loops.md#training-across-rounds) |

## Start here

- **Compare three kinds of extra computation.** Trace a [Coconut latent step](docs/language-model-loops.md#intermediate-steps-before-an-answer), a [LLaDA denoising step](docs/language-model-loops.md#editing-and-denoising-text) and a [Self-Refine revision](docs/language-model-loops.md#feedback-retrieval-and-search). Each changes a different part of the answer-generation process.
- **Understand the architecture.** [Universal Transformers](https://arxiv.org/abs/1807.03819) shows how a shared update can refine every token’s representation across depth. Its adaptive variant lets individual positions stop at different steps.
- **Try changing depth on one checkpoint.** [Huginn’s model card](https://huggingface.co/tomg-group-umd/huginn-0125) gives inference examples and identifies the recurrent core, input layers, and output layers. Vary the recurrence count while keeping the prompt and decoding settings fixed.
- **Interpret a scaling claim.** [How Much Is One Recurrence Worth?](https://arxiv.org/abs/2604.21106) compares tied and untied models at the same executed depth. In one reported configuration, a 410M-parameter model with four recurrences matched the validation loss of a 580M untied model at roughly the training cost of a 1B untied model. Parameter savings and compute savings answer different questions.

## Models and weights

Checkpoint links open the publisher’s file tree. Each entry was checked for public weight files on 2026-11-26. License labels come from the individual model cards; “unspecified” means the card metadata does not declare a weight license. For sharded releases, the file tree includes the index and individual shards.

### Shared-depth models

| Model family | What it offers | Official weight files | Paper or report | Weight license |
| --- | --- | --- | --- | --- |
| ALBERT | A fixed-depth encoder with cross-layer parameter sharing. The base-v2 release is a compact reference for studying weight tying in masked language modeling. | [Base v2](https://huggingface.co/albert/albert-base-v2/tree/main) | [ALBERT](https://arxiv.org/abs/1909.11942) | Apache-2.0 |
| Huginn | A 3.5B language model with a recurrent core between distinct input and output stacks. The same checkpoint supports different recurrence counts at inference. | [Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125/tree/main) | [Recurrent depth report](https://arxiv.org/abs/2502.05171) | Apache-2.0 |
| Ouro | Shared-weight language models with a learned exit mechanism. Base and Thinking releases are available at both sizes. | [1.4B base](https://huggingface.co/ByteDance/Ouro-1.4B/tree/main) · [1.4B Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/tree/main) · [2.6B base](https://huggingface.co/ByteDance/Ouro-2.6B/tree/main) · [2.6B Thinking](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/tree/main) | [Ouro report](https://arxiv.org/abs/2510.25741) | Apache-2.0 |
| LoopCoder-v2 | A 7B instruction-tuned coding model. The downloadable variant runs its 14 shared layers twice. | [7B two-loop checkpoint](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2/tree/main) | [Loop-count study](https://arxiv.org/abs/2606.18023) | Apache-2.0 |
| Nanbeige4.2 | A shared layer stack is reused during a forward pass. The report covers the base model and post-training for tool use and coding. The reported 3B count excludes embeddings. | [3B base](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base/tree/main) · [3B post-trained](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/main) | [Nanbeige4.2 technical report](https://arxiv.org/abs/2607.22083) | Apache-2.0 |
| ARLoop | Uses attention over earlier layer states to decide what each layer reads. The authors also release a 0.6B looped baseline with standard pre-normalization for a same-size residual-design comparison. | [0.6B](https://huggingface.co/osunlp/ARLoop-0.6B/tree/main) · [0.6B baseline](https://huggingface.co/osunlp/ARLoop-baseline-0.6B/tree/main) · [1.2B](https://huggingface.co/osunlp/ARLoop-1.2B/tree/main) | [ARLoop report](https://arxiv.org/abs/2611.05234) | Apache-2.0 |
| Dory 1.0 | NVIDIA’s normalized Transformer has distinct input, recurrent, and output blocks. Its last-loop KV-cache mode keeps one set of keys and values across the recurrent passes. | [4B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/tree/main) · [4B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/tree/main) | [Dory report](https://arxiv.org/abs/2611.03998) | OpenMDW-1.1 |
| Retrofitted TinyLlama | Two conversions of the same pretrained backbone with different target training recurrence counts. These are separately trained models. | [4 recurrences](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-4/tree/main) · [32 recurrences](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-32/tree/main) | [Retrofitting study](https://arxiv.org/abs/2511.07384) | Apache-2.0 |
| LoopUS | Converts Qwen3 into an encoder, shared reasoning block, and decoder. A confidence head can terminate the recurrent stage early. | [Qwen3-1.7B retrofit](https://huggingface.co/Thrillcrazyer/Qwen3_1.7B_LoopUS/tree/main) | [LoopUS paper](https://arxiv.org/abs/2605.11011) | Apache-2.0 |
| LoopFormer | Three models pair different block counts and iteration counts for the same nominal depth of 24. Shortcut modulation allows shorter inference trajectories. | [1 × 24](https://huggingface.co/armenjeddi/LoopFormer-1block-24iterations/tree/main) · [2 × 12](https://huggingface.co/armenjeddi/LoopFormer-2block-12iterations/tree/main) · [3 × 8](https://huggingface.co/armenjeddi/LoopFormer-3block-8iterations/tree/main) | [LoopFormer paper](https://arxiv.org/abs/2602.11451) | Unspecified |
| HRM | Two recurrent attention modules run at different rates. Each task has its own checkpoint; the files are named `checkpoint`. | [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2/tree/main) · [Sudoku Extreme](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme/tree/main) · [Maze-Hard](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard/tree/main) | [HRM paper](https://arxiv.org/abs/2506.21734) | Unspecified |
| HRM-Text | Applies fast and slow recurrent modules to text generation. This release uses a task-completion objective on instruction–response pairs. | [1B original release](https://huggingface.co/sapientinc/HRM-Text-1B/tree/main) | [HRM-Text report](https://arxiv.org/abs/2605.20613) | Apache-2.0 |
| LoopVL | Refines a joint vision–language state through module-level and model-level loops. The authors expose the two recurrence counts for inference. | [LoopVL-1B](https://huggingface.co/TierFlow/LoopVL/tree/main) | [LoopVL paper](https://arxiv.org/abs/2609.38426) | Unspecified |
| RD-VLA | Repeats a weight-tied Transformer in the action head. Each file tree includes the action head and its associated model components for one LIBERO evaluation suite. | [Spatial, 40k steps](https://huggingface.co/hqfang/12_24-24_24_Spatial_40k/tree/main) · [Long, 75k steps](https://huggingface.co/hqfang/12_24-24_24_Long_75k/tree/main) | [RD-VLA paper](https://arxiv.org/abs/2602.07845) | Apache-2.0 |

### Sequence recurrence and iterative decoding

| Model family | State or output updated | Official weight files | Paper | Weight license |
| --- | --- | --- | --- | --- |
| RWKV-4 | Updates the recurrent state once per input token. | [169M Pile](https://huggingface.co/RWKV/rwkv-4-169m-pile/tree/main) | [RWKV](https://arxiv.org/abs/2305.13048) | Unspecified |
| Mamba | Updates a selective state-space representation as the sequence advances. | [130M base, original format](https://huggingface.co/state-spaces/mamba-130m/tree/main) | [Mamba](https://arxiv.org/abs/2312.00752) | Apache-2.0 |
| MDLM | Repeatedly denoises a partially masked sequence. | [OpenWebText checkpoint](https://huggingface.co/kuleshov-group/mdlm-owt/tree/main) | [MDLM](https://arxiv.org/abs/2406.07524) | Apache-2.0 |
| LLaDA | Repeatedly fills masked response positions conditioned on the prompt. | [8B Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/main) | [LLaDA](https://arxiv.org/abs/2502.09992) | MIT |

## Research papers

This section focuses on Looped Transformers. The [language model bibliography](docs/language-model-loops.md) covers other loop mechanisms. The [extended Transformer index](docs/research-and-code.md) covers early translation systems, algorithmic constructions, puzzle solvers, training methods, multimodal applications, and inference engines.

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

### Converting pretrained models and sharing experts

- [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384) — Uses a curriculum to introduce recurrence into pretrained Llama, TinyLlama, and OLMo models. The math experiments compare the retrofit with continued training of the original model under a compute budget. [Training code](https://github.com/mcleish7/retrofitting-recurrence)
- [LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models](https://arxiv.org/abs/2605.11011) — Splits a pretrained model into three stages, gates the input to the recurrent stage, and supervises intermediate depths during post-training. [Code](https://github.com/Thrillcrazyer/LoopUS)
- [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451) — Trains on different trajectory lengths and conditions shared blocks on time and step size. The authors release standard and looped baselines alongside the proposed models. [Baseline collection](https://huggingface.co/collections/armenjeddi/loopformer)
- [MoEUT: Mixture-of-Experts Universal Transformers](https://arxiv.org/abs/2405.16039) — Uses sparse experts in shared attention and feed-forward layers. This increases the number of available parameters while limiting the computation spent on any one token. [Implementation](https://github.com/robertcsordas/moeut)

### Vision, multimodal reasoning, and control

- [Sliced Recursive Transformer](https://arxiv.org/abs/2111.05297) — Reuses vision Transformer layers for image classification. Sliced group self-attention reduces the attention cost of the repeated passes. [Code and model zoo](https://github.com/szq0214/SReT)
- [LoopViT: Scaling Visual ARC with Looped Transformers](https://arxiv.org/abs/2602.02156) — Repeats a hybrid convolution-and-attention block on visual ARC tasks and uses prediction entropy to decide when to stop. [Code](https://github.com/WenjieShu/LoopViT)
- [ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/abs/2604.09168) — Trains intermediate loop counts to reproduce the deepest configuration in image and video generation. The paper compares quality at different inference budgets after a single training run.
- [LoopVL: Recurrent Visual Intelligence](https://arxiv.org/abs/2609.38426) — Studies where visual attention moves as recurrent computation proceeds on the same image and question. The repository documents the loop schedules used for its visual-computation experiments. [Experiment guides](https://github.com/Tier-Flow/LoopVL)
- [Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning](https://arxiv.org/abs/2602.07845) — Varies the number of refinements before predicting a robot action. A convergence-based stopping rule lets easy and difficult manipulation decisions use different compute budgets. [Code and suite-specific weights](https://github.com/rd-vla/rd-vla)
- [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) — Pairs a slower high-level state with a faster low-level state. Its task-specific puzzle checkpoints offer a small-scale setting for tracking successive updates. [Training and evaluation](https://github.com/sapientinc/HRM)

### Inference cost and interpreting hidden states

- [Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models](https://arxiv.org/abs/2510.14961) — Overlaps the refinement of several token positions. The authors evaluate a parallel sampler on a pretrained recurrent-depth model without additional tuning.
- [LoopVQ: Sub 1-bit KV Cache Quantization for Looped Transformers](https://arxiv.org/abs/2611.05304) — Jointly quantizes keys and values across loop steps. The experiments compare cache compression on Ouro and Huginn, including long-context retrieval. [Implementation](https://github.com/mscheong01/LoopVQ)
- [Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer](https://arxiv.org/abs/2507.02199) — Probes Huginn on arithmetic with different readouts. The authors find limited evidence for decodable intermediate arithmetic steps, and the result depends on the probe and layer.
- [On the Chain-of-Thought Monitorability of Looped Language Models](https://arxiv.org/abs/2610.02741) — Varies Ouro’s depth and compares against non-looped models matched by parameters or depth. Some cue-use stress tests become harder to monitor at larger depths; the matched-model comparison has no uniform effect.
- [Fractal basins trap latent reasoning](https://arxiv.org/abs/2609.04963) — Perturbs latent states and maps which solution each trajectory reaches. The authors relate long reasoning times on puzzle tasks to trajectories that linger near almost-correct solutions.

## Technical reports and implementations

| Release | What to look for in the report | Author resources |
| --- | --- | --- |
| [Huginn recurrent-depth report](https://arxiv.org/abs/2502.05171) | How the recurrent block is pretrained and how performance changes as more compute is spent per token at inference. | [Training and evaluation code](https://github.com/seal-rg/recurrent-pretraining) · [Training dataset](https://huggingface.co/datasets/tomg-group-umd/huginn-dataset) |
| [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) | Ouro’s pretraining recipe and the objective used to learn when to exit. Its model card explains the runtime settings that enable adaptive exit. | [Project](https://ouro-llm.github.io/) · [Model card and exit settings](https://huggingface.co/ByteDance/Ouro-1.4B) |
| [Nanbeige4.2-3B technical report](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/Nanbeige42_report.pdf) | How a looped backbone fits into a complete 28T-token pretraining and agent post-training pipeline. The final agent benchmarks reflect that whole pipeline. | [Architecture implementation](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/modeling_nanbeige.py) |
| [LoopCoder-v2 report](https://arxiv.org/abs/2606.18023) | Why cross-loop position offsets and cache sharing matter for parallel looping, and how the authors tested the cost of another pass. | [Implementation](https://github.com/CSJianYang/LoopCoder) · [Released configuration](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2/blob/main/config.json) |
| [ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals](https://arxiv.org/abs/2611.05234) | Sliding-window versus full-history attention across depth, followed by larger pretraining runs. The released baseline is useful when comparing residual designs. | [Pretraining, inference, and evaluation](https://github.com/OSU-NLP-Group/ARLoop) |
| [Dory1.0-4B: A Looped Normalized-GPT](https://arxiv.org/abs/2611.03998) | How loop count, precision, and KV caching interact in a 4B nGPT. The report distinguishes the training loop schedule from the inference-time choice. | [Runtime configuration](https://huggingface.co/docs/transformers/en/model_doc/dory) |
| [HRM-Text: Efficient Pretraining Beyond Scaling](https://arxiv.org/abs/2605.20613) | How hierarchical recurrence, normalization, and instruction–response pretraining fit together. The data and objective also change when comparing to conventional pretrained baselines. | [Training framework](https://github.com/sapientinc/HRM-Text) |
| [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384) | Which backbone is converted and how the recurrence curriculum is scheduled. The checkpoint collection includes untrained conversions and multiple trained depths. | [Full checkpoint collection](https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence) |
| [LoopVL: Recurrent Visual Intelligence](https://arxiv.org/abs/2609.38426) | How language pretraining, multimodal training, and post-training are staged for a shared vision–language state. | [Publisher model card](https://huggingface.co/TierFlow/LoopVL) |

## Analysis tools

- [MoR routing notebook](https://github.com/raymin0223/mixture_of_recursions/blob/main/notebooks/250727_get_mor_routing_decision.ipynb) — Inspect expert-choice routing decisions for individual tokens in a Mixture-of-Recursions model.
- [Huginn latent probes](https://github.com/wenquanlu/huginn-latent-cot) — Compare Logit Lens and Coda Lens readouts at intermediate recurrent depths on arithmetic examples.
- [Loopscape](https://github.com/GilpinLab/loopscape) — Reproduce perturbation and basin-of-attraction analyses on recurrent reasoners using the authors’ experiment code.

## Related lists

- [Awesome Loop Models](https://github.com/huskydoge/Awesome-Loop-Models) — A broader catalog covering looped architectures, implicit models, theory, and applications, with an interactive paper browser.
- [HJSang’s Awesome Looped Transformers](https://github.com/HJSang/awesome-looped-transformers) — A complementary reading list covering recurrent-depth Transformers, systems work, and commentary.

See the [contribution guide](CONTRIBUTING.md) to suggest a paper, checkpoint, or report.
