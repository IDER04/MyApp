# VAN Studio

A Streamlit dashboard for comparing the **VAN** (French acronym for net present
value, NPV) of two investment projects.

The app models two projects side by side and updates the analysis live as the
discount rate, initial outlay, project life, or annual cash flows change.

## Features

- Two independent project models with flat or year-by-year net cash flows
- VAN comparison at a shared discount rate
- Decision verdict based on the `VAN > 0` rule
- Responsive VAN bar chart and discount-rate sensitivity curve
- Cumulative discounted cash flow and annual cash flow charts
- IRR, profitability index, and discounted payback metrics
- Crossover rate where the ranking between the two projects reverses
- CSV export of the comparison

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed in the terminal (usually <http://localhost:8501>).

## Tests

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

## Financial conventions

All cash flows are net cash flows and occur at the end of each year. The initial
investment is made at `t = 0`. Both projects use the same discount rate unless
you incorporate project-specific risk into the cash-flow estimates.

```text
VAN = -I₀ + Σ [CFₜ / (1 + r)ᵗ]
```

- A project is acceptable when its VAN is positive.
- Between mutually exclusive projects, the one with the highest positive VAN is
  recommended.
- The crossover rate is the positive discount rate at which both VANs are equal.
- IRR assumes reinvestment at the IRR and can be unreliable for non-conventional
  cash-flow patterns; the VAN remains the primary metric.
- Different project lives and scales are not adjusted here. For a rigorous
  comparison in those cases, consider a common-life or equivalent annual annuity
  approach.

The tool is for decision support and education, not financial advice.

## Project layout

```text
app.py                          Streamlit interface and charts
finance.py                      Financial calculations
tests/test_finance.py           Unit tests for the financial engine
tests/test_app.py               Streamlit smoke test
.streamlit/config.toml          Theme and app configuration
requirements.txt                Runtime dependencies
requirements-dev.txt            Test dependencies
```

Default project values and colours are defined near the top of `app.py`.
