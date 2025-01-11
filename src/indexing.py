import pandas as pd
import json
from collections import defaultdict

def build_inverted_index(dataframe):
    inverted_index = defaultdict(list)

    for idx, row in dataframe.iterrows():
        doc_id = idx
        words = row['processed_content'].split()

        for word in words:
            if doc_id not in inverted_index[word]:
                inverted_index[word].append(doc_id)

    return inverted_index

if __name__ == "__main__":
    # Φόρτωσε τα προεπεξεργασμένα δεδομένα
    df = pd.read_csv("data/cleaned_wikipedia_articles.csv")

    # Δημιουργία ανεστραμμένου ευρετηρίου
    inverted_index = build_inverted_index(df)

    # Αποθήκευση ευρετηρίου σε αρχείο JSON
    with open("data/inverted_index.json", "w") as json_file:
        json.dump(inverted_index, json_file)

    print("Η δημιουργία του ανεστραμμένου ευρετηρίου ολοκληρώθηκε!")
    print("Το ευρετήριο αποθηκεύτηκε στο 'data/inverted_index.json'")
