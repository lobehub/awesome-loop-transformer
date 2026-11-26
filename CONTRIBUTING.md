# Contributing

Include work that studies a repeated operation in a language model or a system built around one. Describe one full iteration in concrete terms. What goes in, what changes, what carries forward, and what determines when the loop ends? Link to the [mechanism index](docs/language-model-loops.md) for examples.

Place work on shared layers, recurrent depth and their applications in the [Looped Transformer section](docs/research-and-code.md). The wider index covers recurrence over a sequence, intermediate reasoning, iterative decoding, feedback and search, and training across rounds. Ordinary autoregressive decoding or optimizer steps qualify when the work studies that iteration itself. For methods spanning multiple categories, list the paper under its main contribution and cross-link when helpful.

- **Paper:** Link the publisher, OpenReview, or arXiv record. Describe the question and identify the baseline behind any numerical claim.
- **Model or weights:** Link the publisher’s model card and weight files. Name the variant, such as base, instruction-tuned, or experimental control. Verify that the file tree contains weights or weight shards and report the license declared for that checkpoint.
- **Technical report:** Add the release to [reports and tools](docs/reports-and-tools.md). Link the authors’ report and, where available, the architecture implementation, training recipe, or evaluation harness.

When a paper describes an unreleased model, list the paper and identify any available artifacts. Use the [models and weights catalog](docs/models-and-weights.md) for downloadable releases. For efficiency claims, say which quantity the experiment holds fixed, such as unique parameters, executed depth, training FLOPs, or inference budget.
