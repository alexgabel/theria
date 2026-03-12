#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

README_QUICK_LINK_TARGETS = [
    Path("docs/STATUS.md"),
    Path("experiments/phase12/README.md"),
    Path("docs/phase12_for_colleagues.md"),
    Path("docs/phase12_results_snapshot.md"),
]

RECOMMENDED_DOC_PAGES = [
    Path("README.md"),
    Path("docs/STATUS.md"),
    Path("docs/phase12_for_colleagues.md"),
    Path("docs/phase12_results_snapshot.md"),
    Path("experiments/phase12/README.md"),
    Path("examples/README.md"),
]


def _read(path: Path) -> str:
    return (REPO_ROOT / path).read_text()


def _check_files_exist(errors: list[str]) -> None:
    for rel_path in README_QUICK_LINK_TARGETS:
        if not (REPO_ROOT / rel_path).exists():
            errors.append(f"missing required file: {rel_path}")


def _check_readme_links(errors: list[str]) -> None:
    readme = _read(Path("README.md"))
    for rel_path in README_QUICK_LINK_TARGETS:
        if str(rel_path) not in readme:
            errors.append(f"README.md is missing quick-link entry for: {rel_path}")


def _iter_fenced_blocks(text: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    in_block = False
    lang = ""
    block_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("```"):
            if in_block:
                blocks.append((lang, block_lines))
                in_block = False
                lang = ""
                block_lines = []
            else:
                in_block = True
                lang = line[3:].strip().lower()
            continue
        if in_block:
            block_lines.append(line)
    return blocks


def _check_no_cuda_graph_flag_in_runnable_docs(errors: list[str]) -> None:
    for rel_path in RECOMMENDED_DOC_PAGES:
        text = _read(rel_path)
        for lang, block_lines in _iter_fenced_blocks(text):
            if lang not in {"", "bash", "sh", "shell", "console"}:
                continue
            block = "\n".join(block_lines)
            if "CUDA_GRAPH_STATIC=1" in block:
                errors.append(
                    f"{rel_path} contains CUDA_GRAPH_STATIC=1 inside a runnable code block"
                )


def main() -> int:
    errors: list[str] = []
    _check_files_exist(errors)
    _check_readme_links(errors)
    _check_no_cuda_graph_flag_in_runnable_docs(errors)

    if errors:
        print("phase12_docs_check=FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("phase12_docs_check=OK")
    for rel_path in README_QUICK_LINK_TARGETS:
        print(f"- verified: {rel_path}")
    print("- verified: no recommended doc page contains CUDA_GRAPH_STATIC=1 in runnable code blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
