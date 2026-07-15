# Data directory

Real datasets are used **offline, once, only to calibrate** the simulation
(see `src/dynpricing/calibration/calibrate.py`). The agents never train on this
data, and the true demand function is never derived from it directly.

Place raw files here (all optional — calibration falls back to synthetic
defaults if a file is absent):

| Dataset | Expected filename | Licence | Used for |
|---------|-------------------|---------|----------|
| UCI Online Retail II | `online_retail_II.xlsx` or `online_retail_II.csv` | CC BY 4.0 | price band, base demand, elasticity |
| ONS Retail Sales Index | `ons_retail_sales.csv` | Open Government Licence | seasonal amplitude |
| UK Bank Holidays | fetched live via `--holidays` | Open Government Licence | holiday demand effects |
| Amazon Product Dataset (Kaggle) | (optional, not yet wired) | confirm before use | price/cost grounding |

## Sources

* UCI Online Retail II — https://doi.org/10.24432/C5CG6D
* ONS Retail Sales Index — https://www.ons.gov.uk/datasets/retail-sales-index
* UK Bank Holidays — https://www.gov.uk/bank-holidays

This directory is git-ignored except for this README.
