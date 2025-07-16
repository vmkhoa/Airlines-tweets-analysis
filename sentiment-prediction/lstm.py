import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import clean_text
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load GloVe embeddings (100D GloVe pre-trained vectors)
def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Load the GloVe embeddings
glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')  

path = "/Users/khoavan/Downloads/tweets.csv"
df = pd.read_csv(path)

# Label Encoding for Sentiments 
encoder = LabelEncoder()
df['sentiment_encoded'] = encoder.fit_transform(df['sentiment'])

# Convert text to embedding vectors using pre-trained GloVe embeddings
def text_to_embedding(text, embeddings, embedding_dim=100, max_seq_len=10):
    tokens = clean_text(text).split()  # Tokenize cleaned text into words
    embedding_matrix = []
    
    for token in tokens:
        if token in embeddings:
            embedding_matrix.append(embeddings[token])
        else:
            embedding_matrix.append(np.zeros(embedding_dim))  # Zero vector for words not in GloVe
    
    # Ensure the sequence is padded/truncated to the fixed sequence length
    while len(embedding_matrix) < max_seq_len:  # Adjust max_seq_len to whatever max sequence length you want
        embedding_matrix.append(np.zeros(embedding_dim))  # Padding with zero vectors
    
    return np.array(embedding_matrix)[:max_seq_len]

# Convert all texts to embedding sequences
X = np.array([text_to_embedding(text, glove_embeddings) for text in df['text']])
y = df['sentiment_encoded'].values  # Sentiment labels

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for Batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Define the LSTM Model for Sentiment Analysis (without Keras)
class SentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  
        self.fc = nn.Linear(hidden_dim, output_dim)  


    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden_state = h_n[-1]  # Use last hidden state
        self.dropout = nn.Dropout(0.5)
        out = self.fc(last_hidden_state)
        return out

# Hyperparameters
input_dim = 100  # Length of the embedding vector (100D GloVe embeddings)
hidden_dim = 64  # Number of neurons in the hidden layer
output_dim = 4  # Number of sentiment classes (POSITIVE, NEGATIVE, NEUTRAL, MIXED)

# Initialize model, loss function, and optimizer
model = SentimentModel(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()  # Multi-class classification loss

# Training Loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Zero gradients before each step
        outputs = model(X_batch)  # Forward pass
        loss = loss_fn(outputs, y_batch)  # Compute the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Evaluate the Model on Test Data
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)  # Get predicted sentiment
        y_pred.extend(predicted.cpu().numpy())  # Append predictions
        y_true.extend(y_batch.cpu().numpy())  # Append true labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
# Print classification report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_true, y_pred))
# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()