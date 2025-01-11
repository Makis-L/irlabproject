import json
import math

def load_inverted_index(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def process_query(query):
    # Κανονικοποίηση του ερωτήματος
    query = query.lower()
    tokens = query.split()
    return tokens

def boolean_search(query_tokens, inverted_index, operation="AND"):
    results = []
    for token in query_tokens:
        if token in inverted_index:
            results.append(set(inverted_index[token]))
        else:
            results.append(set())
    
    if operation == "AND":
        return set.intersection(*results)
    elif operation == "OR":
        return set.union(*results)
    elif operation == "NOT":
        return set.difference(results[0], *results[1:])
    return set()

def calculate_tf_idf(doc_id, term, inverted_index, total_docs, doc_lengths):
    term_frequency = inverted_index[term].count(doc_id) if term in inverted_index else 0
    document_frequency = len(inverted_index[term]) if term in inverted_index else 0

    tf = term_frequency / doc_lengths[doc_id]
    idf = math.log(total_docs / (1 + document_frequency))

    return tf * idf

def rank_results(query_tokens, results, inverted_index, total_docs, doc_lengths):
    scores = {}
    for doc_id in results:
        scores[doc_id] = sum(
            calculate_tf_idf(doc_id, term, inverted_index, total_docs, doc_lengths)
            for term in query_tokens if term in inverted_index
        )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    # Φόρτωσε το ανεστραμμένο ευρετήριο
    inverted_index = load_inverted_index("data/inverted_index.json")
    total_docs = len(inverted_index)
    doc_lengths = {doc_id: len(content.split()) for doc_id, content in enumerate(inverted_index.values())}

    # Ερώτημα Χρήστη
    query = input("Εισάγετε το ερώτημά σας: ")
    query_tokens = process_query(query)

    # Boolean Search
    operation = input("Επιλέξτε λειτουργία (AND, OR, NOT): ").upper()
    results = boolean_search(query_tokens, inverted_index, operation)

    # Κατάταξη Αποτελεσμάτων
    ranked_results = rank_results(query_tokens, results, inverted_index, total_docs, doc_lengths)

    # Εμφάνιση Αποτελεσμάτων
    print("\nΑποτελέσματα Αναζήτησης:")
    for doc_id, score in ranked_results:
        print(f"Έγγραφο: {doc_id}, Βαθμολογία: {score}")
