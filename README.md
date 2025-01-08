# Text Doxument Classification using NLP, RNN and LSTM.🤖

## This Project aims to classify the document category using a given text to classify if the text is related to politics, tech, business or sports.

![Application](https://github.com/sahermuhamed1/Text-Document-Classifier/blob/main/src/Application.png)


# Dataset📊

The dataset contains a csv file that has the text string and the label for it

# Text Preprocessing

I use NLP to classify the label for the given text.

Text Preprocessing Steps:
- *Text Cleaning*: remove the unneeded characters and convert the string to a lowercase
- *Remove Stopwords*
- *Tokenize the Text*
- *Stemming and Lemmitization*

# NLP 

I utilized several models to evaluate the performance of each model and take the best one:
1. *Count Vectorizer + Naive Bayes*
2. *Count Vectorizer with N-Grams + Naive Bayes*
3. *tfidf vectorizer + Naive Bayes*

# LSTM

I utilize a LSTM model  I utilized an LSTM model to classify text documents into different categories. The model achieved an accuracy of 0.92 on the test set. The model was trained for 25 epochs with a batch size of 20. The model consists of an embedding layer, two LSTM layers, and two dense layers. The model was trained using the Adam optimizer with a learning rate of 0.001. The model was evaluated using the sparse categorical cross-entropy loss function and the accuracy metric. The model achieved an accuracy of 0.92 on the test set. The model was evaluated using the classification report and confusion matrix. The classification report provides a detailed summary of the model's performance on the test set, including precision, recall, and F1-score for each class. The confusion matrix provides a visual representation of the model's performance on the test set, showing the number of true positives, true negatives, false positives, and false negatives for each class. The model performed well on the test set, achieving an accuracy of 0.92 and demonstrating good performance across all classes.

steps I've done:
1. *Tokenization*
2. *Padding*
3. *Label Encoding*
4. *Model Building*
5. *Model Training*
6. *Model Evaluation*

# Deployment 🛠️

In this project, we utilize Streamlit as our deployment tool to transform our AI model into an interactive dashboard. This platform enables users to input text, triggering our AI model's text classification algorithm, providing a classification for it.


# Usage🤔

To use the trained emotion detection model:

1. Load the trained model from the provided pickle file.
2. Preprocess the input text by removing non-letter characters, converting words to lowercase, and removing stopwords.
3. Use the trained model to predict the emotion of the preprocessed text.
Example code:
```python
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Preprocess the input text
input_text = "South Africa's Schalk Burger was named player of the year as the Tri-Nations champions swept the top honours"
preprocessed_text = preprocess_text(input_text)

# Predict the label of the input text
prediction = model.predict([preprocessed_text])[0]

print("Predicted Emotion:", prediction)
```


# Contact info📩
For inquiries or further collaboration, please contact Saher Mohammed at [sahermuhamed176@gmail.com].🥰

[Saher's Linkdin](https://www.linkedin.com/in/sahermuhamed/)

