import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import joblib
from real_news import fetch_news 
from fake_news import scrape_fake_claims

nltk.download('stopwords')

true_news = fetch_news(query="India",page_size=100)  
fake_news = scrape_fake_claims()
print(f"True news dataset size: {true_news.shape}")
print(f"Fake news dataset size: {fake_news.shape}")


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocessing(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove multiple spaces and strip leading/trailing spaces
    text = text.lower()  # Convert to lowercase
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

# def fix_title(text):
#     if pd.isna(text) or len(text.split()) < 3:  
#         return ''
#     try:
#         return text.encode('latin1').decode('utf-8', errors='ignore')
#     except (UnicodeEncodeError, UnicodeDecodeError):
#         return text 

# true_news['title'] = true_news['title'].fillna('').apply(fix_title)
# fake_news['title'] = fake_news['title'].fillna('').apply(fix_title)

# max_text_length = 300  
# true_news['short_text'] = true_news['text'].apply(lambda x: x[:max_text_length])
# fake_news['short_text'] = fake_news['text'].apply(lambda x: x[:max_text_length])

# true_news['content'] = true_news['title'] + " " + true_news['short_text']
# fake_news['content'] = fake_news['title'] + " " + fake_news['short_text']
 
true_news['cleaned_content'] = true_news['content'].apply(preprocessing)
fake_news['cleaned_content'] = fake_news['content'].apply(preprocessing)

fake_news['label'] = 1
true_news['label'] = 0

data = pd.concat([fake_news[['cleaned_content', 'label']], true_news[['cleaned_content', 'label']]], ignore_index=True)

X_train, X_test, Y_train, Y_test = train_test_split(data["cleaned_content"], data["label"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, alpha=0.0001, random_state=42, class_weight='balanced')
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, y_pred)}")
print(classification_report(Y_test, y_pred))

joblib.dump(model, 'fake_news_detector.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

#Model testing
# def test_news(news):
#     cleaned_news = preprocessing(news)

#     transformed_news = vectorizer.transform([cleaned_news]).toarray()

#     prediction = model.predict(transformed_news)

#     if prediction[0] == 1:
#         print("The news is likely Fake.")
#     else:
#         print("The news is likely Real.")

# sample_news = """
# New York governor questions the constitutionality of federal tax overhaul
# """
# test_news(sample_news)
