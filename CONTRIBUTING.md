# Contributing

You can [suggest a resource](https://github.com/lobehub/awesome-loop-transformer/issues/new/choose) or submit a pull request directly. For a suggestion, send the primary source and a sentence explaining what repeats. For a correction, link the existing entry and the source that supports the change.

## What qualifies

A reader should be able to trace one iteration. What operation repeats? What state reaches the next pass? What sets the pass count or stops the loop? For example, Ouro-Thinking 2.6B passes its hidden states through the same 48 decoder blocks four times, for 192 block applications per token. [Architecture and configuration](docs/model-architectures.md#five-decoder-architectures)

Shared layers, recurrent depth and their applications belong in the [Looped Transformer bibliography](docs/research-and-code.md). The [wider language-model index](docs/language-model-loops.md) covers memory across segments, intermediate reasoning, iterative decoding, feedback, search and training across rounds. Ordinary autoregressive decoding and optimizer steps qualify when the work specifically studies those iterations.

## What reviewers check

| Submission | Evidence to include |
| --- | --- |
| Paper or preprint | Link the title to the original paper. Give the year of first public release and a sentence explaining the research question or comparison. Keep quantitative claims tied to the task, baseline and measured budget. |
| Model or checkpoint | Add the named variant to [models and weights](docs/models-and-weights.md). Link the publisher’s file tree containing the weights or shards and cite the model card, config or report for the loop. Label adapters, quantizations and controls. Use the checkpoint’s declared license, or mark it *Unspecified* when the publisher gives none. |
| Technical report, implementation or tool | Add the authors’ report or repository to [reports and tools](docs/reports-and-tools.md). Describe what readers can inspect or run, such as the recurrence schedule, training recipe or evaluation harness. |
| Architecture figure | Add it to the [figure guide](docs/model-architectures.md) and link it from the matching model entry. Credit the creator, link the original figure or config, and describe the recurrent path in the caption or alt text. Link out to a third-party figure when permission to reproduce it is unclear. |

For efficiency claims, distinguish unique parameters, block applications, training FLOPs and inference cost. If a study compares accuracy at matched parameter counts, report that comparison. A speed claim needs a timing measurement. A paper describing an unreleased checkpoint can still be included and its release status stated.

## Opening a pull request

1. Search the catalog by title, arXiv ID or model family. Update an existing record for a new paper version or checkpoint variant. Place a new work under its main contribution and cross-link other relevant sections.
2. Edit the topic page under `docs/` in English, following the ordering and format of its neighboring entries. For papers, use both surnames for two authors and the first surname plus “et al.” for three or more. Add the source BibTeX record with the full author list to [references.bib](references.bib) once per work.
3. Regenerate the homepage from the repository root with `python3 scripts/update_readme.py`. Include the resulting README changes and open the links you added.
4. Open a pull request against `main`. Identify the changed entry and the source a reviewer should use to verify it. A figure number, config field or checkpoint file path makes that review easier. The **Catalog checks** workflow verifies the generated README, local links and bibliography. Run `python3 scripts/check_bibliography.py` locally to check for duplicate citation keys and paper entries missing a BibTeX record.
