# Contributing to Compitum

Thanks for your interest in improving Compitum! This guide helps you set up your environment, make focused changes, and open high‑quality pull requests.

## Quick Start

- Fork and clone the repo
- Create a virtual environment and install dev deps:
  ```bash
  python -m venv .venv && . .venv/bin/activate
  pip install -e ".[dev]"
  ```
- Run lint and tests locally:
  ```bash
  ruff check .
  pytest -q -m "not routerbench"
  ```

## RouterBench (optional)
Some integration tests and evaluation scripts depend on RouterBench.
- Use a separate venv to avoid conflicts:
  ```bash
  python -m venv .venv-routerbench && . .venv-routerbench/bin/activate
  pip install -r src/routerbench/requirements.txt
  ```
- Fetch the 5‑shot dataset (not committed to Git):
  ```bash
  python scripts/fetch_routerbench.py --also-copy-to-src
  ```
- Run optional tests marked `routerbench`:
  ```bash
  python -m pytest -q -m routerbench
  ```

## Development Workflow
- Create a topic branch from `main`
- Keep commits small and descriptive
- Add/update tests next to the code you change
- Update docs/README if behavior or usage changes

## Commit Hygiene
- Do not commit large binaries or datasets (e.g., `.pkl`). Use the fetch script or release assets
- Do not commit secrets or tokens; rotate immediately if leaked
- Prefer clear naming and small, well‑scoped functions

## PR Checklist
- [ ] `ruff check .` passes
- [ ] `pytest -q -m "not routerbench"` passes
- [ ] Docs or README updated (if applicable)
- [ ] No large artifacts or secrets included
- [ ] Screenshots or short notes for UI/docs changes (if helpful)

## Security Considerations

Our security philosophy mirrors our core principles: we favor continuous monitoring and adaptive boundaries over rigid, static rules. We distinguish between the production-grade **Core Engine (`src/compitum`)** and the less-strict **Research and Benchmarking code (`src/routerbench`, `benchmarks/`)**.

When contributing, please help us build a more secure system:

*   **Be vigilant with the Core Engine.** Treat all user input as untrusted.
*   **Understand the context.** A vulnerability in the core is more severe than one in a research script.
*   **Engage with us.** Security is a shared responsibility. If you see a potential issue, please raise it.
*   **Research and Benchmarking (`src/routerbench`, `benchmarks/`):** This code is for research and evaluation. It has lower security requirements than the core engine. For example, it uses `pickle` for data serialization, which is not a secure practice for production code. **If you are concerned about potential vulnerabilities in this code, consider running it in an isolated environment (e.g., a Docker container or a dedicated virtual machine).**

For more details, see our [Security Policy](SECURITY.md).

## Code of Conduct & Security
- By participating you agree to the project’s [Code of Conduct](CODE_OF_CONDUCT.md)
- Report vulnerabilities privately to: paulctiffany@gmail.com (see [SECURITY.md](SECURITY.md))

## License
By contributing, you agree that your contributions will be licensed under this repository’s license.

