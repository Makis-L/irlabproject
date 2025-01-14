import json
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from collections import Counter
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from evaluation import evaluate_retrieval

# Κατέβασμα των απαραίτητων δεδομένων
nltk.download('punkt')
nltk.download('stopwords')

# Φόρτωση δεδομένων
def load_data():
    with open('data/inverted_index.json', 'r', encoding='utf-8') as file:
        inverted_index = json.load(file)

    with open('data/cleaned_wikipedia_articles.json', 'r', encoding='utf-8') as file:
        processed_articles = json.load(file)
    
    return inverted_index, processed_articles

# Προεπεξεργασία ερωτήματος
def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^a-z\s]', '', query)
    tokens = word_tokenize(query)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Boolean αναζήτηση με υποστήριξη AND, OR, NOT
def boolean_search(query, index, processed_articles, operation="AND"):
    tokens = preprocess_query(query)
    print(f"Query Tokens: {tokens}")
    
    if not tokens:
        return set()

    if operation == "AND":
        results = set(index.get(tokens[0], []))
        for token in tokens[1:]:
            results = results.intersection(set(index.get(token, [])))
    elif operation == "OR":
        results = set()
        for token in tokens:
            results = results.union(set(index.get(token, [])))
    elif operation == "NOT":
        results = set(range(len(processed_articles)))
        for token in tokens:
            results = results.difference(set(index.get(token, [])))
    else:
        raise ValueError("Unsupported operation. Use 'AND', 'OR', or 'NOT'.")
    
    return results

# Υπολογισμός cosine similarity για VSM
def vsm_search(query, index, documents):
    query_tokens = preprocess_query(query)
    print(f"VSM Query Tokens: {query_tokens}")

    # Υπολογισμός IDF
    N = len(documents)
    idf = {token: math.log((N + 1) / (len(index.get(token, [])) + 1)) + 1 for token in query_tokens}

    # Δημιουργία διανυσμάτων για κάθε έγγραφο
    document_vectors = []
    for doc_id, doc in enumerate(documents):
        doc_vector = [doc['processed_content'].split().count(token) * idf.get(token, 0) for token in query_tokens]
        document_vectors.append(doc_vector)

    # Δημιουργία διανύσματος ερωτήματος
    query_vector = [idf.get(token, 0) for token in query_tokens]

    # Υπολογισμός cosine similarity
    similarities = []
    for doc_id, doc_vector in enumerate(document_vectors):
        dot_product = np.dot(query_vector, doc_vector)
        norm_query = np.linalg.norm(query_vector)
        norm_doc = np.linalg.norm(doc_vector)
        similarity = dot_product / (norm_query * norm_doc) if norm_query and norm_doc else 0
        similarities.append((doc_id, similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Επεξεργασία δεδομένων για BM25
def prepare_bm25_data(documents):
    return [doc['processed_content'].split() for doc in documents]

# Αναζήτηση με Okapi BM25
def bm25_search(query, documents):
    tokenized_docs = prepare_bm25_data(documents)
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = preprocess_query(query)
    print(f"BM25 Query Tokens: {query_tokens}")
    scores = bm25.get_scores(query_tokens)
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

# Διεπαφή αναζήτησης
def search_interface():
    inverted_index, processed_articles = load_data()
    print("Welcome to the Search Engine!")
    print("Enter your query (or type 'exit' to quit):")

    while True:
        query = input("\n> ").strip()
        if query.lower() == 'exit':
            print("Exiting the search engine. Goodbye!")
            break

        print("\nChoose retrieval method:")
        print("1. Boolean Retrieval (AND, OR, NOT)")
        print("2. Vector Space Model (VSM)")
        print("3. Okapi BM25")
        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == "1":
            print("\nAvailable Boolean operations: AND, OR, NOT")
            operation = input("Enter Boolean operation: ").strip().upper()
            if operation not in {"AND", "OR", "NOT"}:
                print("Invalid operation. Please try again.")
                continue
            results = boolean_search(query, inverted_index, processed_articles, operation)
            print("\nTop Results:")
            for doc_id in results:
                print(f"- {processed_articles[doc_id]['title']}")
        elif choice == "2":
            results = vsm_search(query, inverted_index, processed_articles)
            print("\nTop Results:")
            for doc_id, score in results[:5]:  # Display top 5
                print(f"- {processed_articles[doc_id]['title']} (Score: {score:.4f})")
        elif choice == "3":
            results = bm25_search(query, processed_articles)
            print("\nTop Results:")
            for doc_id, score in results[:5]:  # Display top 5
                print(f"- {processed_articles[doc_id]['title']} (Score: {score:.4f})")
        else:
            print("Invalid choice. Please try again.")
            continue

# Εκκίνηση διεπαφής
if __name__ == "__main__":
    search_interface()
