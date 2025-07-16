# Airlines tweets analysis
## 1. Setup
To install the environment of this project, create a virtual environment and activate, then install the dependencies in **requirements.txt**.  
The file **lstm.py** in **sentiment-prediction** needs the [GloVe](https://github.com/stanfordnlp/GloVe) embedding. To download, do:  
`curl -O http://nlp.stanford.edu/data/glove.6B.zip` then `unzip glove.6B.zip`.
## 2. Overview
This project seeks to analyze a collection of airline tweets. The goal is to predict sentiments, find trends, and gain statistical insights from this dataset. The link to the dataset is here: https://drive.google.com/file/d/1ZzhTKdesTC2c5t5EUURUvI7MdDiqxrs2/view.  
The dataset collects information related to tweets commenting on flight experience. An essential column is sentiment, which curiously has 4 labels: POSITIVE, NEGATIVE, NEUTRAL, and MIXED. 
_ For data analysis and sentiment prediction, see **analysis.ipynb** notebook.   
_ For topic modeling, go to **topic-modeling/src/main.py**, the configuration is in **config.yaml** in **topic-modeling**, the dataset with topics column is in **outputs**.  
_ For charts, see the **charts** folder in **visualization**.  
_ For deep learning methods, see **sentiment-prediction**.  
## 3. Methods
Most machine learning methods, in spite of vectorization choice, yield an accuracy of no higher than 78%. They all struggle to meaningfully predict the "MIXED" sentiment label, due to imbalance. The best method is training a DistilBERT model on the dataset. Accuracy improved significanty (82%) and the model achieved some meaningful metrics in predicting "MIXED".  
As for topic modeling, the result isn't too meaningful.  
TODO: Find a better method and evalution for topic modeling.








