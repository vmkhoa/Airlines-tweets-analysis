import pandas as pd
import re
from tqdm import tqdm
tqdm.pandas()

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a Pandas dataframe."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return pd.DataFrame()

def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, remove links, punctuation, etc."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text
