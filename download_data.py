from pathlib import Path
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
OUT = Path("data/titanic.csv")

OUT.parent.mkdir(parents=True, exist_ok=True)
data = pd.read_csv(DATA_URL)
data.to_csv(OUT, index=False)
print(f"Saved {len(data)} rows to {OUT}")
