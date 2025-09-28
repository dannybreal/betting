from pathlib import Path
path = Path(r'c:/Users/dan67/Nuance/betting/NEW/streamlit_app.py')
text = path.read_text()
old = "                actual = row.get(\"Actual\")\n                predicted = row.get(\"Predicted Winner\")\n                if pd.isna(actual) or pd.isna(predicted):\n                    return [\"\" for _ in row]\n                if actual == predicted:\n                    return [\"background-color: rgba(34, 197, 94, 0.18)\"] * len(row)"
new = "                actual = row.get(\"Actual\")\n                predicted = row.get(\"Predicted Winner\")\n                if not actual or not predicted or pd.isna(predicted):\n                    return [\"\" for _ in row]\n                if actual == predicted:\n                    return [\"background-color: rgba(34, 197, 94, 0.18)\"] * len(row)"
if old not in text:
    raise SystemExit('style block not found')
text = text.replace(old, new, 1)
path.write_text(text)
