import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Κατέβασμα απαραίτητων δεδομένων του NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Αρχικοποίηση των εργαλείων
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Συνάρτηση για καθαρισμό και επεξεργασία κειμένου
def preprocess_text(text):
    
    # Μετατροπή σε πεζά
    text = text.lower()
    # Αφαίρεση ειδικών χαρακτήρων και αριθμών
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenization
    tokens = word_tokenize(text)
    # Αφαίρεση stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return " ".join(stemmed_tokens)

# Φόρτωση των δεδομένων από το JSON αρχείο
def load_articles(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Αποθήκευση δεδομένων σε JSON αρχείο
def save_articles(articles, file_path):

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)

# Κύρια συνάρτηση προεπεξεργασίας
def preprocess_articles(input_file, output_file):

    articles = load_articles(input_file)
    for article in articles:
        article["processed_content"] = preprocess_text(article["content"])
    save_articles(articles, output_file)
    print(f"Η προεπεξεργασία ολοκληρώθηκε! Τα δεδομένα αποθηκεύτηκαν στο {output_file}")

if __name__ == "__main__":
    input_file = "data/wikipedia_articles.json"
    output_file = "data/cleaned_wikipedia_articles.json"
    preprocess_articles(input_file, output_file)
