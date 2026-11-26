# Reports, implementations and analysis tools

[Overview](../README.md) · [Selected readings](../README.md#selected-readings) · [BibTeX](../references.bib)

## Contents

- [Technical reports and implementations](#technical-reports-and-implementations)
- [Analysis tools](#analysis-tools)
- [Related lists](#related-lists)

## Technical reports and implementations

The links in the research focus column lead to papers on the same question.

| Release report | Research focus | Author resources |
| --- | --- | --- |
| <a id="report-huginn"></a>[Huginn recurrent-depth report](https://arxiv.org/abs/2502.05171) | How the recurrent block is pretrained and [how performance changes with inference depth](research-and-code.md#depth-allocation-and-scaling). | [Training and evaluation code](https://github.com/seal-rg/recurrent-pretraining) · [Training dataset](https://huggingface.co/datasets/tomg-group-umd/huginn-dataset) |
| <a id="report-ouro"></a>[Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) | How Ouro [supervises predictions at successive depths and trains an exit gate](research-and-code.md#training-across-recurrent-depth). The model card documents [how gate scores select an output depth](research-and-code.md#depth-allocation-and-scaling). | [Project](https://ouro-llm.github.io/) · [Model card and exit settings](https://huggingface.co/ByteDance/Ouro-1.4B) |
| <a id="report-nanbeige"></a>[Nanbeige4.2-3B technical report](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/Nanbeige42_report.pdf) | How a looped backbone fits into a [28T-token pretraining and agent post-training pipeline](research-and-code.md#training-across-recurrent-depth). The final agent benchmarks reflect that whole pipeline. | [Architecture implementation](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/main/modeling_nanbeige.py) |
| <a id="report-loopcoder-v2"></a>[LoopCoder-v2 report](https://arxiv.org/abs/2606.18023) | Why cross-loop position offsets and cache sharing matter for parallel looping, and how the authors tested [the cost of another pass](research-and-code.md#depth-allocation-and-scaling). | [Implementation](https://github.com/CSJianYang/LoopCoder) · [Released configuration](https://huggingface.co/Multilingual-Multimodal-NLP/LoopCoder-V2/blob/main/config.json) |
| <a id="report-arloop"></a>[ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals](https://arxiv.org/abs/2611.05234) | [Sliding-window versus full-history attention across depth](research-and-code.md#training-across-recurrent-depth), followed by larger pretraining runs. The released baseline is useful when comparing residual designs. | [Pretraining, inference, and evaluation](https://github.com/OSU-NLP-Group/ARLoop) |
| <a id="report-dory"></a>[Dory1.0-4B: A Looped Normalized-GPT](https://arxiv.org/abs/2611.03998) | How [inference depth](research-and-code.md#depth-allocation-and-scaling), precision and [KV caching](research-and-code.md#inference-attention-and-memory) interact in a 4B nGPT. The report distinguishes the training loop schedule from the inference-time setting. | [Runtime configuration](https://huggingface.co/docs/transformers/en/model_doc/dory) |
| <a id="report-hrm-text"></a>[HRM-Text: Efficient Pretraining Beyond Scaling](https://arxiv.org/abs/2605.20613) | How [fast and slow recurrent modules](research-and-code.md#puzzles-and-recurrent-state) and normalization fit into [instruction–response pretraining](research-and-code.md#training-across-recurrent-depth). The data and objective also change when comparing to conventional pretrained baselines. | [Training framework](https://github.com/sapientinc/HRM-Text) |
| <a id="report-retrofitting"></a>[Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/abs/2511.07384) | How the authors [introduce recurrence into a pretrained backbone](research-and-code.md#converting-pretrained-models) and schedule the training curriculum. The checkpoint collection includes untrained conversions and multiple trained depths. | [Full checkpoint collection](https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence) |
| <a id="report-loopvl"></a>[LoopVL: Recurrent Visual Intelligence](https://arxiv.org/abs/2609.38426) | How language pretraining, multimodal training, and post-training are staged for a [shared vision–language state](research-and-code.md#vision-speech-control-and-tabular-data). | [Publisher model card](https://huggingface.co/TierFlow/LoopVL) |

## Analysis tools

- [MoR routing notebook](https://github.com/raymin0223/mixture_of_recursions/blob/main/notebooks/250727_get_mor_routing_decision.ipynb) — Inspect expert-choice routing decisions for individual tokens in a Mixture-of-Recursions model.
- [Huginn latent probes](https://github.com/wenquanlu/huginn-latent-cot) — Compare Logit Lens and Coda Lens readouts at intermediate recurrent depths on arithmetic examples.
- [Loopscape](https://github.com/GilpinLab/loopscape) — Reproduce perturbation and basin-of-attraction analyses on recurrent reasoners using the authors’ experiment code.

## Related lists

- [Awesome Loop Models](https://github.com/huskydoge/Awesome-Loop-Models) — A broader catalog covering looped architectures, implicit models, theory, and applications, with an interactive paper browser.
- [HJSang’s Awesome Looped Transformers](https://github.com/HJSang/awesome-looped-transformers) — A complementary reading list covering recurrent-depth Transformers, systems work, and commentary.
