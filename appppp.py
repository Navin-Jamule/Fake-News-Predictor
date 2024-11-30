import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('stopwords')

st.title("Fake News Detection with Analysis")
st.write("An app to detect if the news is fake or real using machine learning.")
news_df = pd.read_csv('train.csv.zip')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
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
st.write(f"Train Accuracy: {train_accuracy:.2f}")
st.write(f"Test Accuracy: {test_accuracy:.2f}")

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
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    add_article(input_text)
    if pred == 1:
        st.write('The News is **Fake**')
    else:
        st.write('The News is **Real**')

if recent_articles:
    st.sidebar.subheader("Recently Analyzed Articles")
    for article in recent_articles:
        st.sidebar.write(article)

st.write("### Insights")
st.write("Let's take a look at some data insights:")

st.write(f"Total dataset size: {len(news_df)}")
label_distribution = news_df['label'].value_counts()
st.write(f"Label distribution:\n {label_distribution}")

st.write("#### Distribution of Fake and Real News")
fig, ax = plt.subplots()
sns.countplot(news_df['label'], ax=ax)
ax.set_xticklabels(['Real', 'Fake'])
st.pyplot(fig)

st.write("#### Classification Report")
st.text(classification_report(Y_test, y_pred_test))

st.write("#### Confusion Matrix")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred_test)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
st.pyplot(fig)

st.sidebar.title("Export Analyzed Articles")
if st.sidebar.button('Download Article History'):
    recent_articles_df = pd.DataFrame(recent_articles, columns=['Article'])
    recent_articles_df.to_csv('recent_articles.csv', index=False)
    st.sidebar.write("Download ready!")

st.write("### Preprocessed Sample")
sample_idx = np.random.choice(len(news_df), size=1)[0]
st.write("Original Content:", news_df['content'].iloc[sample_idx])
st.write("After Preprocessing and Stemming:", stemming(news_df['content'].iloc[sample_idx]))

st.write("### Top TF-IDF Features")
feature_names = vector.get_feature_names_out()
sorted_items = np.argsort(vector.idf_)[:10]
top_features = [feature_names[i] for i in sorted_items]
st.write(f"Top 10 TF-IDF features: {top_features}")

st.write("### Explore Word Frequencies in Fake and Real News")
fake_news = news_df[news_df['label'] == 1]
real_news = news_df[news_df['label'] == 0]

fake_text = ' '.join(fake_news['content'])
real_text = ' '.join(real_news['content'])

from wordcloud import WordCloud

st.write("#### Word Cloud for Fake News")
fig, ax = plt.subplots()
fake_wc = WordCloud(width=800, height=400, background_color='black').generate(fake_text)
ax.imshow(fake_wc, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

st.write("#### Word Cloud for Real News")
fig, ax = plt.subplots()
real_wc = WordCloud(width=800, height=400, background_color='black').generate(real_text)
ax.imshow(real_wc, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)
