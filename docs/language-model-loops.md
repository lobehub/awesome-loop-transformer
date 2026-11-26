# Loops in language models

[Overview](../README.md) · [Selected readings](../README.md#selected-readings) · [BibTeX](../references.bib)

Follow the state from one iteration to the next. With a fixed prompt, an equilibrium model can refine one hidden state many times. A recurrent sequence model instead reads new input at each step. The sections below distinguish the state being updated and the time scale of the update. Years refer to first public release. [Full citations in BibTeX](../references.bib)

## Contents

- [Recurrent depth and equilibrium](#recurrent-depth-and-equilibrium)
- [Memory across tokens and segments](#memory-across-tokens-and-segments)
- [Intermediate steps before an answer](#intermediate-steps-before-an-answer)
- [Editing and denoising text](#editing-and-denoising-text)
- [Feedback, retrieval and search](#feedback-retrieval-and-search)
- [Training across rounds](#training-across-rounds)

## Recurrent depth and equilibrium

- Bai et al. (2019). [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) — A fixed-point solver repeatedly evaluates a shared transformation until its state settles or the solver reaches its budget. The paper tests an equilibrium self-attention model on language modeling. Read alongside the fixed-pass and adaptive-depth Transformers in the main list. [Code](https://github.com/locuslab/deq) · [WikiText-103 checkpoint](models-and-weights.md#model-deq-transformer)

For layer sharing, recurrent pretraining, halting and deployment, continue to the [reading path](../README.md#selected-readings) and the [Transformer bibliography](research-and-code.md).

## Memory across tokens and segments

- Mikolov et al. (2010). [Recurrent neural network based language model](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html) — Feeds the previous hidden state and the current word into a recurrent layer to predict the next word. This gives a simple baseline for asking what a model keeps as text advances.
- Dai et al. (2019). [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) — Each segment can attend to hidden activations cached from the preceding segment. The current segment produces the memory available to the next one. [Code](https://github.com/kimiyoung/transformer-xl)
- Hutchins et al. (2022). [Block-Recurrent Transformers](https://arxiv.org/abs/2203.07852)<a id="paper-block-recurrent-transformers"></a> — A Transformer cell processes one block of tokens at a time and updates a set of gated state vectors for the next block. The authors compare long-context language modeling perplexity with Transformer-XL on books, papers and source code. [Original implementation (archived)](https://github.com/google-research/meliad#block-recurrent-transformer)
- Bulatov et al. (2022). [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) — A Transformer reads a small set of memory tokens along with a segment and writes new memory embeddings. The next segment receives those embeddings, so the memory budget can stay fixed as the input grows. [Code](https://github.com/booydar/recurrent-memory-transformer)
- Peng et al. (2023). [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) — RWKV’s recurrent inference updates a compact state for each new token. Its training formulation can process sequence positions in parallel. [Code](https://github.com/BlinkDL/RWKV-LM) · [RWKV-4 169M Pile checkpoint](models-and-weights.md#model-rwkv-4)
- Sun et al. (2023). [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621) — RetNet presents parallel, recurrent and chunkwise forms of the same retention operation. The recurrent form carries a decayed summary of past keys and values. [Code](https://github.com/microsoft/torchscale)
- Gu and Dao (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Mamba’s selective state-space layer updates a running state with input-dependent dynamics. The sequence supplies the steps, and the current token influences which information persists. [Code](https://github.com/state-spaces/mamba) · [130M base checkpoint](models-and-weights.md#model-mamba)
- Sun et al. (2024). [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620) — A TTT layer uses the weights of a small predictor as its sequence state. A self-supervised update on incoming data changes the predictor that later positions use. [Code](https://github.com/test-time-training/ttt-lm-pytorch)
- Huang et al. (2026). [Latent Feedback Is a Compute and Data Amplifier](https://arxiv.org/abs/2611.20404) — Each token passes its top-layer state to the next token’s input. The authors compare feedback with training an ordinary Transformer for more epochs on the same text, varying model size, unique data and compute. [Code](https://github.com/ifm-ai/LoRM/tree/main/works/latent_feedback) · [Feedback model and matched control](models-and-weights.md#model-latent-feedback)
- Huang et al. (2026). [Trading Depth for Time in Recurrent Transformers](https://arxiv.org/abs/2609.21605)<a id="paper-trading-depth"></a> — Inserts a latent step between successive vocabulary tokens in a Transformer with cross-token feedback. The experiments compare this step with doubling physical depth and with repeating layers. Each configuration runs 2L blocks per decoded vocabulary token, where L is the original backbone depth. [LRT codebase](https://github.com/OoDBag/Latent_recurrent_transformer)

## Intermediate steps before an answer

- Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — In few-shot chain-of-thought prompting, worked examples elicit intermediate text before the answer. Each reasoning token becomes part of the prefix for subsequent predictions.
- Goyal et al. (2023). [Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226) — Adds learned pause tokens during training and inference. Additional positions let the causal model transform its representations before reading out an answer.
- Zelikman et al. (2024). [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — Quiet-STaR samples short rationales at token positions during training and learns from how well those rationales help predict subsequent text. Its intermediate thoughts are discrete token sequences. [Code](https://github.com/ezelikman/quiet-star)
- Hao et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) — Coconut feeds the last hidden state directly into the model as its next input embedding for a sequence of latent steps, then resumes text generation. [Code](https://github.com/facebookresearch/coconut)
- Chen and Fu (2026). [Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs](https://arxiv.org/abs/2609.01117) — A small recurrent network repeatedly revises a proposed sequence of latent thoughts. A frozen language model consumes those vectors to produce the answer.

## Editing and denoising text

- Ghazvininejad et al. (2019). [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324) — The translation decoder predicts masked target positions in parallel, masks its least confident predictions and tries them again. The number of refinement rounds controls decoding cost. [Code](https://github.com/facebookresearch/Mask-Predict)
- Gu et al. (2019). [Levenshtein Transformer](https://arxiv.org/abs/1905.11006) — A translation decoder edits a draft through learned deletions and insertions. The sequence can change length between iterations.
- Sahoo et al. (2024). [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) — MDLM trains a masked denoiser and samples text by repeatedly filling positions in a partially masked sequence. The paper connects its objective to masked language modeling. [Code](https://github.com/kuleshov-group/mdlm) · [OpenWebText checkpoint](models-and-weights.md#model-mdlm)
- Nie et al. (2025). [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) — LLaDA starts with masked response positions and repeatedly predicts tokens using the prompt and the partially filled answer. The released sampler exposes the step count and remasking strategy. [Code](https://github.com/ML-GSAI/LLaDA) · [LLaDA-8B-Instruct checkpoint](models-and-weights.md#model-llada)
- Arriola et al. (2025). [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) — Block diffusion advances left to right across blocks and runs a denoising loop inside each block. Completed blocks provide context for the next one. [Code](https://github.com/kuleshov-group/bd3lms)
- Arriola and Kuleshov (2026). [Set Diffusion: Interpolating Token Orderings Between Autoregression and Diffusion for Fast and Flexible Decoding](https://arxiv.org/abs/2607.01775) — Set diffusion extends block decoding to token sets with flexible positions and sizes. The sampler can choose a different ordering and update its KV cache after each step. [Code](https://github.com/kuleshov-group/setdlms)

## Feedback, retrieval and search

- Madaan et al. (2023). [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) — Self-Refine calls the model to critique a draft, then provides the draft and feedback to a revision call. Its task experiments compare successive drafts with the first answer. [Code](https://github.com/madaan/self-refine)
- Huang et al. (2023). [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) — Tests intrinsic self-correction on reasoning problems. Asking models to reconsider an answer using only their own feedback sometimes lowers accuracy in the evaluated settings.
- Yao et al. (2022). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — ReAct interleaves reasoning, actions and observations. A tool or environment returns information that the model uses to decide its next action. [Code](https://github.com/ysymyth/ReAct)
- Trivedi et al. (2022). [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509) — IRCoT uses an intermediate reasoning sentence to retrieve evidence. Retrieved passages then condition the next reasoning step on multi-hop questions. [Code](https://github.com/StonyBrookNLP/ircot)
- Yao et al. (2023). [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Tree of Thoughts maintains multiple partial solutions. The language model proposes and evaluates continuations, and a search procedure selects which branches to expand. [Code](https://github.com/princeton-nlp/tree-of-thought-llm)
- Shinn et al. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) — Reflexion writes language feedback from an attempt into episodic memory. A later trial reads that reflection before acting again. [Code](https://github.com/noahshinn/reflexion)

## Training across rounds

- Zelikman et al. (2022). [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — STaR generates rationales, retries failed examples with the correct answer supplied, and fine-tunes on rationales that produce correct answers. The updated model generates the next round. [Code](https://github.com/ezelikman/STaR)
- Gulcehre et al. (2023). [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998) — ReST generates a batch with the current policy, scores its outputs and reuses the offline data for policy improvement. The paper studies this grow-and-improve procedure on machine translation.
- Chen et al. (2024). [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335) — SPIN uses responses from the previous model alongside human demonstrations to train the next model. The updated model becomes the source of synthetic responses in a subsequent round. [Code](https://github.com/uclaml/SPIN)
