import pandas as pd
import numpy as np

from pathlib import Path


DATA_DIR = Path("data/")
## print(Path.cwd())

K2 = pd.read_csv(DATA_DIR / "K2.csv", sep = "," , comment = "#")
## print(K2.head(5))

KEPLER = pd.read_csv(DATA_DIR / "KEPLER.csv", sep = "," , comment = "#")
## print(KEPLER.head(5))

TESS = pd.read_csv(DATA_DIR / "TESS.csv", sep = "," , comment = "#")
## ;print(TESS.head(5))

## Start work with kepler data
print(KEPLER.columns)
Kepler = KEPLER.drop(columns=["kepid","kepoi_name","kepler_name"])
