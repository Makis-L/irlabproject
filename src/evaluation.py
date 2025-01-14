# Αξιολόγηση Απόδοσης

def evaluate_retrieval(relevant_docs, retrieved_docs):

    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_docs)
    
    # Precision
    true_positives = len(relevant_set & retrieved_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    
    # Recall
    recall = true_positives / len(relevant_set) if relevant_set else 0
    
    # F1-Score
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall else 0
    
    return {"Precision": precision, "Recall": recall, "F1-Score": f1_score}

# Παράδειγμα Χρήσης
if __name__ == "__main__":
    # Παράδειγμα δεδομένων
    relevant_docs = [0, 1, 2]  # IDs των σχετικών άρθρων
    retrieved_docs = [1, 2, 3]  # IDs των ανακτηθέντων άρθρων

    # Υπολογισμός Μετρικών
    metrics = evaluate_retrieval(relevant_docs, retrieved_docs)
    print("Μετρικές Αξιολόγησης:")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
