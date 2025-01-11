import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Κατέβασε τα απαραίτητα datasets (μία φορά)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Κανονικοποίηση (lowercasing)
    text = text.lower()

    # Αφαίρεση ειδικών χαρακτήρων
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Αφαίρεση stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Συνένωση tokens σε κείμενο
    processed_text = ' '.join(tokens)
    return processed_text

if __name__ == "__main__":
    # Φόρτωσε τα δεδομένα
    df = pd.read_csv("data/wikipedia_articles.csv")

    # Επεξεργασία του περιεχομένου
    df['processed_content'] = df['content'].apply(preprocess_text)

    # Αποθήκευση των προεπεξεργασμένων δεδομένων
    df.to_csv("data/cleaned_wikipedia_articles.csv", index=False)
    print("Η προεπεξεργασία ολοκληρώθηκε! Τα δεδομένα αποθηκεύτηκαν στο 'data/cleaned_wikipedia_articles.csv'")
