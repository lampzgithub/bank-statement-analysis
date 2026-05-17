# Statement PDF Analyzer

Streamlit dashboard that parses a bank statement PDF and shows:

- money came from transfers
- money came from refunds
- money went to transfers
- money paid via UPI

It also includes inflow/outflow totals, category charts, and full transaction table.

## Run

```bash
streamlit run app.py
```

## Notes

- Parser currently targets statement layout similar to Bank of Maharashtra PDF tables.
