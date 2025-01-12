import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Κατέβασε τα απαραίτητα δεδομένα NLTK (μία φορά)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Ορισμός εξατομικευμένων stopwords
custom_stopwords = set(stopwords.words('english')).union({"ml", "ai", "data"})

def preprocess_text(text):
    # Έλεγχος αν το κείμενο είναι έγκυρο
    if not isinstance(text, str) or len(text) < 10:
        return ""
    
    # Κανονικοποίηση (lowercase)
    text = text.lower()

    # Αφαίρεση ειδικών χαρακτήρων
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Αφαίρεση stopwords
    tokens = [word for word in tokens if word not in custom_stopwords]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Επανασύνδεση tokens σε κείμενο
    return ' '.join(tokens)

if __name__ == "__main__":
    # Φόρτωση δεδομένων
    input_file = "data/wikipedia_articles.csv"
    output_file = "data/cleaned_wikipedia_articles.csv"

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Το αρχείο {input_file} δεν βρέθηκε.")
        exit()

    # Εφαρμογή προεπεξεργασίας
    df['processed_content'] = df['content'].apply(preprocess_text)

    # Αφαίρεση κενών εγγραφών
    df = df[df['processed_content'] != ""]

    # Αποθήκευση δεδομένων
    df.to_csv(output_file, index=False)
    print(f"Η προεπεξεργασία ολοκληρώθηκε. Τα δεδομένα αποθηκεύτηκαν στο {output_file}")
