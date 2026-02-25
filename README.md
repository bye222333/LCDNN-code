# LCDNN 

This repository contains the code of LCDNN
## Contents

- `scripts/lcdnn_paper_debug_fix.py`: main runnable script (exported from Colab notebook)
- `notebooks/LCDNN_paper_debug_fix.ipynb`: Jupyter notebook version
- `requirements.txt`: Python dependencies
- `data/`, `outputs/`: placeholders for datasets / generated results

## Quickstart (Python)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/lcdnn_paper_debug_fix.py
```

## Notes

- Some baselines in the script require **optional** packages (e.g., `torch`, `causal-learn`, `networkx`, `causalassembly`).  
  They are listed as optional in `requirements.txt`.

## Reproducibility tips

Inside `scripts/lcdnn_paper_debug_fix.py`, look for flags like:

- `RUN_SYNTHETIC`
- `RUN_CAUSALASSEMBLY`
- `MODE = "FAST"` vs `"PAPER"`

Toggle them as needed for your experiments.
