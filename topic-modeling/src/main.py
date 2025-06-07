import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from tqdm import tqdm
import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import clean_text
tqdm.pandas()

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df['cleaned'] = df['text'].progress_apply(clean_text)  # show progress bar
    return df

def run_nmf(docs, n_topics=5, n_words=10):
    """
    Apply Non-Negative Matrix Factorization (NMF) to a collection of text documents
    to uncover hidden topics and assign each document to its most relevant topic.

    Parameters:
    docs (list or pd.Series): Cleaned text documents.
    n_topics (int): Number of latent topics to extract.
    n_words (int): Number of top words to return for each topic.

    Returns:
    topics (list of tuples): Each tuple contains a topic index and its top representative words.
    topic_assignments (list of int): Index of the most relevant topic for each document.
    """
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    # TF-IDF matrix: rows = documents, columns = terms
    X = vectorizer.fit_transform(docs)      

    nmf_model = NMF(n_components=n_topics, random_state=42)
    
    # W[i, j] = strength of topic j in document i
    W = nmf_model.fit_transform(X)
    # H[j, k] = importance of term k in topic j
    H = nmf_model.components_
    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for topic_idx, topic in enumerate(H):
        # Get top n_words with highest weight in topic j (from H)
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append((topic_idx, top_words))

    # Assign each document the topic with highest weight in W
    topic_assignments = np.argmax(W, axis=1)

    return topics, topic_assignments

def load_config(path='config.yaml'):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config('config.yaml')
    df = load_and_preprocess(config['data'])
    topics, topic_assignments = run_nmf(
        df['cleaned'], 
        n_topics=config['topics'], 
        n_words=config['words']
        )
    df['topic'] = topic_assignments
    output_path = config['output']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {config['output']}")
    print("\nTop Words per Topic:")
    for topic_idx, words in topics:
        print(f"Topic {topic_idx}: {' | '.join(words)}")

if __name__ == "__main__":
    main()
