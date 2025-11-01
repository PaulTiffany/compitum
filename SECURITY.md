# Security Policy

We take security seriously and appreciate responsible disclosures.

## Reporting a Vulnerability

Please email security reports to: paulctiffany@gmail.com

- Provide a detailed description and steps to reproduce.
- Include any logs, PoCs, or affected version info.
- Do not create a public issue for sensitive reports.

We will acknowledge receipt within 3 business days and aim to provide an initial assessment within 7 business days.

## Supported Versions

- Stable: 0.1.x (security fixes considered)
- Older versions: best‑effort only

## Scope

This policy covers the Compitum codebase and configuration artifacts in this repository. It does not cover third‑party datasets or upstream projects (e.g., RouterBench). For upstream issues, please report directly to their maintainers.

## Known Security Considerations

### Use of Pickle in `routerbench`

The `src/routerbench` submodule, which is used for research and benchmarking, uses the `pickle` module for data serialization. Deserializing data with `pickle` can execute arbitrary code and is only safe with trusted data.

The use of `pickle` in `routerbench` is a known and accepted risk for the intended use case (research with deterministic inputs). However, you should be aware of this and only use the `routerbench` components with data from trusted sources.
