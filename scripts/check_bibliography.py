"""Check that the primary paper links in the two indexes have BibTeX records."""

from pathlib import Path
import re
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
INDEXES = ("docs/research-and-code.md", "docs/language-model-loops.md")
PAPER = re.compile(r"^- .+? \(\d{4}\)\. \[(.+?)\]\((https?://[^)]+)\)")
ENTRY = re.compile(r"^\s*@\w+\s*[{(]\s*([^,\s]+)\s*,", re.MULTILINE)
# The next field or end of the entry bounds each value, including braced titles.
FIELD = re.compile(
    r'(?:^|,)\s*(title|url|eprint)\s*=\s*[{"](.*?)[}"]\s*'
    r'(?=,\s*(?:[\w-]+\s*=|[})])|[})]\s*$)',
    re.IGNORECASE | re.DOTALL,
)


def arxiv_id(url):
    parts = urlsplit(url)
    if parts.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        match = re.fullmatch(r"/(?:abs|pdf|html)/(.+?)(?:v\d+)?(?:\.pdf)?/?", parts.path)
        if match:
            return match.group(1)
    return None


def normalized_title(title):
    return "".join(char for char in title.casefold() if char.isalnum())


def main():
    bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
    entries = list(ENTRY.finditer(bibliography))
    keys, titles, urls, arxiv_ids = set(), set(), set(), set()
    errors = []
    for index, entry in enumerate(entries):
        key = entry.group(1)
        if key.casefold() in keys:
            errors.append(f"references.bib: duplicate citation key {key}")
        keys.add(key.casefold())
        end = entries[index + 1].start() if index + 1 < len(entries) else len(bibliography)
        body = bibliography[entry.end():end]
        fields = {name.lower(): value.strip() for name, value in FIELD.findall(body)}
        if "title" in fields:
            titles.add(normalized_title(fields["title"]))
        if "url" in fields:
            urls.add(fields["url"].rstrip("/"))
            identifier = arxiv_id(fields["url"])
            if identifier:
                arxiv_ids.add(identifier)
        if "eprint" in fields:
            arxiv_ids.add(re.sub(r"v\d+$", "", fields["eprint"]))

    paper_count = 0
    for filename in INDEXES:
        lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            paper = PAPER.match(line)
            if not paper:
                if line.startswith("- ") and "](http" in line:
                    errors.append(f"{filename}:{number}: expected Author (year). [Title](source)")
                continue
            paper_count += 1
            title, url = paper.groups()
            identifier = arxiv_id(url)
            if identifier:
                found = identifier in arxiv_ids
            else:
                found = url.rstrip("/") in urls or normalized_title(title) in titles
            if not found:
                errors.append(f"{filename}:{number}: missing BibTeX record for {title} ({url})")

    if errors:
        sys.exit("\n".join(errors))
    print(f"Matched {paper_count} paper entries against {len(keys)} BibTeX records.")


if __name__ == "__main__":
    main()
