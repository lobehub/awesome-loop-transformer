# Reports, implementations and analysis tools

[Home](../README.md) · [Model weights](models-and-weights.md) · [Papers by topic](research-and-code.md)

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
