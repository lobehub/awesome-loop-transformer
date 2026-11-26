# Contributing

Include work where a Transformer block or layer stack reuses learned parameters across depth. State which components are shared and how the number of passes is chosen. A training method or systems paper fits when it studies that internal recurrence directly.

- **Paper:** Link the publisher, OpenReview, or arXiv record. Describe the question and identify the baseline behind any numerical claim.
- **Model or weights:** Link the publisher’s model card and weight files. Name the variant, such as base, instruction-tuned, or experimental control. Verify that the file tree contains weights or weight shards and report the license declared for that checkpoint.
- **Technical report:** Link the authors’ report and, where available, the architecture implementation, training recipe, or evaluation harness.

When a paper describes an unreleased model, list the paper and identify any available artifacts. Keep the weights table for downloadable releases. For efficiency claims, say which quantity the experiment holds fixed, such as unique parameters, executed depth, training FLOPs, or inference budget.
