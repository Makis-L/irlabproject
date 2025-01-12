import json
from sklearn.metrics import precision_score, recall_score, f1_score

def load_relevant_documents(file_path):
    # Φόρτωσε τα ιδανικά (relevant) έγγραφα για κάθε ερώτημα
    with open(file_path, "r") as file:
        return json.load(file)

def evaluate_query_results(retrieved, relevant):
    # Υπολογισμός Precision, Recall και F1 για ένα ερώτημα
    y_true = [1 if doc_id in relevant else 0 for doc_id in retrieved]
    y_pred = [1] * len(retrieved)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

def mean_average_precision(queries_results, relevant_docs):
    # Υπολογισμός Mean Average Precision (MAP)
    average_precisions = []

    for query, retrieved in queries_results.items():
        relevant = relevant_docs[query]
        num_relevant = 0
        sum_precision = 0

        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                num_relevant += 1
                sum_precision += num_relevant / i

        if relevant:
            average_precisions.append(sum_precision / len(relevant))
        else:
            average_precisions.append(0)

    return sum(average_precisions) / len(average_precisions)

if __name__ == "__main__":
    # Φόρτωσε το ανεστραμμένο ευρετήριο
    inverted_index = json.load(open("data/inverted_index.json"))

    # Δημιούργησε σύνολο δοκιμαστικών ερωτημάτων
    queries_results = {
        "machine learning": [1, 3, 5],
        "data science": [2, 4],
        "artificial intelligence": [3, 5, 6]
    }

    # Φόρτωσε τα ιδανικά σχετικά έγγραφα
    relevant_docs = {
        "machine learning": [1, 5],
        "data science": [2],
        "artificial intelligence": [5, 6]
    }

    # Υπολογισμός μετρικών για κάθε ερώτημα
    for query, retrieved_docs in queries_results.items():
        relevant = relevant_docs.get(query, [])
        precision, recall, f1 = evaluate_query_results(retrieved_docs, relevant)

        print(f"Αποτελέσματα για το ερώτημα: '{query}'")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}\n")

    # Υπολογισμός MAP
    map_score = mean_average_precision(queries_results, relevant_docs)
    print(f"Mean Average Precision (MAP): {map_score:.2f}")
