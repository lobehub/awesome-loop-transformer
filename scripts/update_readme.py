"""Refresh the English README catalogs from their standalone topic pages."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def split_sections(text, level):
    parts = re.split(rf"^{'#' * level} (.+)\n", text, flags=re.MULTILINE)
    return dict(zip(parts[1::2], (body.strip() for body in parts[2::2])))


def sections(filename):
    return split_sections((DOCS / filename).read_text(encoding="utf-8"), 2)


def details(label, body, *, expanded=False):
    tag = "<details open>" if expanded else "<details>"
    return f"{tag}\n<summary>{label}</summary>\n\n{body}\n\n</details>"


def paper_index(filename):
    text = (DOCS / filename).read_text(encoding="utf-8")
    introduction = text.split("\n\n", 3)[2]
    topics = sections(filename)
    contents = topics.pop("Contents")
    parts = [introduction, contents]
    parts.extend(f"### {name}\n\n{body}" for name, body in topics.items())
    return "\n\n".join(parts)


def model_index(figures):
    filename = "models-and-weights.md"
    text = (DOCS / filename).read_text(encoding="utf-8")
    provenance = next(line for line in text.splitlines() if line.startswith("Checkpoint links"))
    topics = sections(filename)
    parts = [provenance, topics.pop("Contents")]
    for name, body in topics.items():
        parts.append(f"### {name}")
        if name == "Release variants":
            for title, variants in split_sections(body, 3).items():
                parts.extend([f"#### {title}", details(f"Show all {title} checkpoints", variants)])
            continue
        parts.append(body)
        # Model names in each table link to the corresponding paper figure.
        slugs = set(re.findall(r"model-architectures\.md#([\w-]+)", body))
        for title, diagram in figures.items():
            if title.lower().replace(" ", "-") in slugs:
                image, commentary = diagram.split("\n\n", 1)
                caption, sources = commentary.rsplit("\n\n", 1)
                parts.extend([f"#### {title}", caption, details(
                    f"Architecture figure and sources for {title}", f"{image}\n\n{sources}",
                    expanded=title in {"Huginn", "LoopVL"},
                )])
    return "\n\n".join(parts)


def home_links(text):
    return (text.replace("](../assets/", "](assets/")
                .replace("](../references.bib)", "](references.bib)")
                .replace("](model-architectures.md#", "](#")
                .replace("](../README.md#", "](#")
                .replace("](research-and-code.md)", "](#looped-transformer-papers)"))


def main():
    figures = sections("model-architectures.md")
    schedule = figures.pop("Five released decoder designs")
    figures.pop("Figures from the papers")
    reports = sections("reports-and-tools.md")
    reports.pop("Contents")
    blocks = {
        "comparison": schedule,
        "models": model_index(figures),
        "papers": paper_index("research-and-code.md"),
        "related": paper_index("language-model-loops.md"),
        "reports": "\n\n".join(f"### {title}\n\n{body}" for title, body in reports.items()),
    }
    paper_counts = [len(re.findall(r"^- .+? \(\d{4}\)\. \[", blocks[key], re.MULTILINE))
                    for key in ("papers", "related")]
    models = sum(line.startswith("| ") and "/tree/main)" in line
                 for line in blocks["models"].splitlines())
    blocks["overview"] = (
        f"The collection includes **{paper_counts[0]} papers on looped Transformers**, "
        f"**{paper_counts[1]} on related language model loops**, and "
        f"**{models} model entries with downloadable weights**."
    )
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for key, content in blocks.items():
        start = f"<!-- catalog:{key} -->"
        end = f"<!-- /catalog:{key} -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"Missing or duplicated README markers for {key}")
        before, rest = text.split(start)
        _, after = rest.split(end)
        text = f"{before}{start}\n\n{home_links(content)}\n\n{end}{after}"
    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
