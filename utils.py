import pandas as pd
import re
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
tqdm.pandas()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


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
    tokens = nltk.word_tokenize(text)            #tokenize
    #Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens if token not in stop_words and len(token) > 2
        ]
    return ' '.join(tokens)


def preprocess(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Apply cleaning to a specified text column with progress bar."""
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' column not found in DataFrame.")
    print(f"Cleaning text column: '{text_column}'")
    df[text_column] = df[text_column].astype(str).progress_apply(clean_text)
    return df