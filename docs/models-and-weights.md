# Models and weights

[Overview](../README.md) · [Selected readings](../README.md#selected-readings) · [BibTeX](../references.bib)

Checkpoint links open the publisher’s file tree. Each entry was checked for public weight files on 2026-11-26. License labels come from the individual model cards; “unspecified” means the card metadata does not declare a weight license. For sharded releases, the file tree includes the index and individual shards.

## Contents

- [Shared-depth models](#shared-depth-models)
- [Sequence recurrence and iterative decoding](#sequence-recurrence-and-iterative-decoding)

## Shared-depth models

| Model family | Recurrent mechanism | Released checkpoints | Reference | Weight license |
| --- | --- | --- | --- | --- |
| ALBERT | A fixed-depth encoder with cross-layer parameter sharing. The base-v2 release is a compact reference for studying weight tying in masked language modeling. | [Base v2](https://huggingface.co/albert/albert-base-v2/tree/main) | [ALBERT](https://arxiv.org/abs/1909.11942) | Apache-2.0 |
| ARLoop | Uses attention over earlier layer states to decide what each layer reads. The authors also release a 0.6B looped baseline with standard pre-normalization for a same-size residual-design comparison. | [0.6B](https://huggingface.co/osunlp/ARLoop-0.6B/tree/main) · [0.6B baseline](https://huggingface.co/osunlp/ARLoop-baseline-0.6B/tree/main) · [1.2B](https://huggingface.co/osunlp/ARLoop-1.2B/tree/main) | [ARLoop report](https://arxiv.org/abs/2611.05234) | Apache-2.0 |
| Dory 1.0 | NVIDIA’s normalized Transformer has distinct input, recurrent, and output blocks. Its last-loop KV-cache mode keeps one set of keys and values across the recurrent passes. | [4B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/tree/main) · [4B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/tree/main) | [Dory report](https://arxiv.org/abs/2611.03998) | OpenMDW-1.1 |
| HRM | Two recurrent attention modules run at different rates. Each task has its own checkpoint; the files are named `checkpoint`. | [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2/tree/main) · [Sudoku Extreme](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme/tree/main) · [Maze-Hard](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard/tree/main) | [HRM paper](https://arxiv.org/abs/2506.21734) | Unspecified |
| HRM-Text | Applies fast and slow recurrent modules to text generation. This release uses a task-completion objective on instruction–response pairs. | [1B original release](https://huggingface.co/sapientinc/HRM-Text-1B/tree/main) | [HRM-Text report](https://arxiv.org/abs/2605.20613) | Apache-2.0 |
| Huginn | A 3.5B language model with a recurrent core between distinct input and output stacks. The same checkpoint supports different recurrence counts at inference. | [Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125/tree/main) | [Recurrent depth report](https://arxiv.org/abs/2502.05171) | Apache-2.0 |
| LoopCoder-v2 | A 7B instruction-tuned coding model. The downloadable variant runs its 14 shared layers twice. | [7B two-loop checkpoint](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2/tree/main) | [Loop-count study](https://arxiv.org/abs/2606.18023) | Apache-2.0 |
| LoopFormer | Three models pair different block counts and iteration counts for the same nominal depth of 24. Shortcut modulation allows shorter inference trajectories. | [1 × 24](https://huggingface.co/armenjeddi/LoopFormer-1block-24iterations/tree/main) · [2 × 12](https://huggingface.co/armenjeddi/LoopFormer-2block-12iterations/tree/main) · [3 × 8](https://huggingface.co/armenjeddi/LoopFormer-3block-8iterations/tree/main) | [LoopFormer paper](https://arxiv.org/abs/2602.11451) | Unspecified |
| LoopUS | Converts Qwen3 into an encoder, shared reasoning block, and decoder. A confidence head can terminate the recurrent stage early. | [Qwen3-1.7B retrofit](https://huggingface.co/Thrillcrazyer/Qwen3_1.7B_LoopUS/tree/main) | [LoopUS paper](https://arxiv.org/abs/2605.11011) | Apache-2.0 |
| LoopVL | Refines a joint vision–language state through module-level and model-level loops. The authors expose the two recurrence counts for inference. | [LoopVL-1B](https://huggingface.co/TierFlow/LoopVL/tree/main) | [LoopVL paper](https://arxiv.org/abs/2609.38426) | Unspecified |
| Nanbeige4.2 | A shared layer stack is reused during a forward pass. The report covers the base model and post-training for tool use and coding. The reported 3B count excludes embeddings. | [3B base](https://huggingface.co/Nanbeige/Nanbeige4.2-3B-Base/tree/main) · [3B post-trained](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/main) | [Nanbeige4.2 technical report](https://arxiv.org/abs/2607.22083) | Apache-2.0 |
| Ouro | Shared-weight language models with a learned exit mechanism. Base and Thinking releases are available at both sizes. | [1.4B base](https://huggingface.co/ByteDance/Ouro-1.4B/tree/main) · [1.4B Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking/tree/main) · [2.6B base](https://huggingface.co/ByteDance/Ouro-2.6B/tree/main) · [2.6B Thinking](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/tree/main) | [Ouro report](https://arxiv.org/abs/2510.25741) | Apache-2.0 |
| RD-VLA | Repeats a weight-tied Transformer in the action head. Each file tree includes the action head and its associated model components for one LIBERO evaluation suite. | [Spatial, 40k steps](https://huggingface.co/hqfang/12_24-24_24_Spatial_40k/tree/main) · [Long, 75k steps](https://huggingface.co/hqfang/12_24-24_24_Long_75k/tree/main) | [RD-VLA paper](https://arxiv.org/abs/2602.07845) | Apache-2.0 |
| Retrofitted TinyLlama | Two conversions of the same pretrained backbone with different target training recurrence counts. These are separately trained models. | [4 recurrences](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-4/tree/main) · [32 recurrences](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-32/tree/main) | [Retrofitting study](https://arxiv.org/abs/2511.07384) | Apache-2.0 |

## Sequence recurrence and iterative decoding

| Model family | Recurrent mechanism | Released checkpoints | Reference | Weight license |
| --- | --- | --- | --- | --- |
| LLaDA | Repeatedly fills masked response positions conditioned on the prompt. | [8B Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tree/main) | [LLaDA](https://arxiv.org/abs/2502.09992) | MIT |
| Mamba | Updates a selective state-space representation as the sequence advances. | [130M base, original format](https://huggingface.co/state-spaces/mamba-130m/tree/main) | [Mamba](https://arxiv.org/abs/2312.00752) | Apache-2.0 |
| MDLM | Repeatedly denoises a partially masked sequence. | [OpenWebText checkpoint](https://huggingface.co/kuleshov-group/mdlm-owt/tree/main) | [MDLM](https://arxiv.org/abs/2406.07524) | Apache-2.0 |
| RWKV-4 | Updates the recurrent state once per input token. | [169M Pile](https://huggingface.co/RWKV/rwkv-4-169m-pile/tree/main) | [RWKV](https://arxiv.org/abs/2305.13048) | Unspecified |
