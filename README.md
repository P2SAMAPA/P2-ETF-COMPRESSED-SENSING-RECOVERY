# Compressed Sensing Recovery Engine for ETFs

Applies compressed sensing (Candès, Romberg, Tao) to ETF return series. Each ETF's returns are assumed sparse in a transform domain (e.g., DCT). Random projections are used to recover the signal via L1 minimisation. The reconstruction error is an anomaly score: high error → non‑sparse / noisy / regime‑shifting.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- DCT transform (sparsifying basis)
- Gaussian random measurement matrix
- L1 recovery via Lasso (scikit‑learn)
- Score = mean squared reconstruction error
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-compressed-sensing-recovery-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n log n) per ETF)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High anomaly score → ETF returns are not well approximated by a sparse representation → likely noisy, regime‑shifting, or anomalous.
- Low anomaly score → ETF returns are structured, predictable, potentially trend‑following.

## Requirements

See `requirements.txt`.
