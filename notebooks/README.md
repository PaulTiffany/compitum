Notebooks
=========

This folder hosts example Jupyter notebooks for Compitum.

Quick start
- Create and activate a virtual environment (optional but recommended).
- Install Jupyter and Compitum in editable mode.
- Launch Jupyter and open the notebook.

Commands
- Windows (PowerShell)
  - python -m venv .venv
  - .venv\Scripts\Activate.ps1
  - pip install --upgrade pip
  - pip install jupyter
  - pip install -e .
  - python -m ipykernel install --user --name compitum-venv --display-name "Python (compitum)"
  - jupyter notebook

- macOS/Linux (bash)
  - python3 -m venv .venv
  - source .venv/bin/activate
  - pip install --upgrade pip
  - pip install jupyter
  - pip install -e .
  - python -m ipykernel install --user --name compitum-venv --display-name "Python (compitum)"
  - jupyter notebook

Notes
- GitHub renders `.ipynb` files directly in the repository UI.
- If running from a subfolder, make sure your working directory is the repo root so `pip install -e .` can find `pyproject.toml`.
- Some examples may reference optional extras (e.g., RouterBench). See `README.md` for setup details.
- New: notebooks/Router_Workflow_Tour.ipynb (lightweight tour; CI-friendly)
