import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from wordcloud import WordCloud

os.makedirs('visualization/charts', exist_ok=True)
df = pd.read_csv('/Users/khoavan/Downloads/tweets.csv')

# Plot sentiment distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
# Save the chart to the 'charts' folder
plt.savefig('visualization/charts/sentiment_distribution.png', bbox_inches='tight')
plt.close()  

# Plot trend by date
df['tweet_created'] = pd.to_datetime(df['tweet_created'], format='%d-%m-%y %H:%M')
df['date'] = df['tweet_created'].dt.date

plt.figure(figsize=(10, 6))
sns.countplot(x='date', hue='sentiment', data=df, palette='Set2')
plt.title('Sentiment Trends By Days')
plt.xlabel('Date')
plt.ylabel('Tweet Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()

# Save 
plt.savefig('visualization/charts/sentiment_trends_by_day.png', bbox_inches='tight')
plt.close()

# Word cloud
all_text = ' '.join(df['text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title('Word Cloud of Most Frequent Words')
plt.savefig('visualization/charts/word_cloud.png', bbox_inches='tight')
plt.close()