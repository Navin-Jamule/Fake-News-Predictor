import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

nltk.download('stopwords')

# Load and preprocess dataset
news_df = pd.read_csv('train.csv.zip')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)

X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)

# Sidebar setup
st.sidebar.title("Article History")
recent_articles = []

def add_article(article):
    if article not in recent_articles:
        recent_articles.append(article)
    if len(recent_articles) > 5:
        recent_articles.pop(0)

st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction_proba = model.predict_proba(input_data)[0]
    return prediction_proba

if input_text:
    pred_proba = prediction(input_text)
    add_article(input_text)
    
    # Display result
    realness_score = pred_proba[0] * 100
    fakeness_score = pred_proba[1] * 100
    
    st.write(f"Trueness Score: {realness_score:.2f}%")
    st.write(f"Fakeness Score: {fakeness_score:.2f}%")
    
    # Display gauge for Trueness
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=realness_score,
        title={'text': "Trueness Percentage"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

if recent_articles:
    st.sidebar.subheader("Recently Analyzed Articles")
    for article in recent_articles:
        st.sidebar.write(article)

# Insights section
st.write("### Insights")

# Train and Test Accuracy side-by-side
st.write("### Train and Test Accuracy")

col1, col2 = st.columns(2)

# Train Accuracy diagram
with col1:
    st.write("#### Train Accuracy")
    fig_train = go.Figure(go.Indicator(
        mode="gauge+number",
        value=train_accuracy * 100,
        title={'text': "Train Accuracy (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "blue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "blue"}
            ]
        }
    ))
    st.plotly_chart(fig_train, use_container_width=True)

# Test Accuracy diagram
with col2:
    st.write("#### Test Accuracy")
    fig_test = go.Figure(go.Indicator(
        mode="gauge+number",
        value=test_accuracy * 100,
        title={'text': "Test Accuracy (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ]
        }
    ))
    st.plotly_chart(fig_test, use_container_width=True)

# Classification Report
st.write("#### Classification Report")
st.text(classification_report(Y_test, y_pred_test))

# Confusion Matrix
st.write("#### Confusion Matrix")
cm = confusion_matrix(Y_test, y_pred_test)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)
