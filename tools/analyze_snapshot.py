import json
from pathlib import Path


def main():
    p = Path("repo_snapshot.jsonl")
    if not p.exists():
        print("repo_snapshot.jsonl not found")
        return 1
    files = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("type") == "file":
                files.append(
                    (
                        j.get("path", ""),
                        int(j.get("estimated_tokens", 0)),
                        int(j.get("size_chars", 0)),
                    )
                )
    files.sort(key=lambda x: x[1], reverse=True)
    for path, tokens, chars in files[:50]:
        print(f"{tokens:>8} tokens  {chars:>9} chars  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
