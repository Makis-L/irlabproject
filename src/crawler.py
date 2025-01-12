import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

def fetch_wikipedia_articles(query, num_articles=10):
    base_url = "https://en.wikipedia.org/wiki/"
    search_url = f"https://en.wikipedia.org/w/index.php?search={query}"
    
    try:
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Σφάλμα κατά τη σύνδεση: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = list(set(a['href'] for a in soup.select('a[href^="/wiki/"]') if ':' not in a['href']))[:num_articles]

    articles = []
    for link in tqdm(links, desc="Fetching articles"):
        url = f"https://en.wikipedia.org{link}"
        try:
            article_response = requests.get(url, timeout=10)
            article_response.raise_for_status()
        except requests.RequestException as e:
            print(f"Αποτυχία ανάκτησης άρθρου: {e}")
            continue

        article_soup = BeautifulSoup(article_response.text, 'html.parser')
        title = article_soup.find('h1').text
        paragraphs = article_soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])

        if len(content) > 100:  # Ελάχιστο όριο λέξεων
            articles.append({"title": title, "content": content, "url": url})

    return articles

if __name__ == "__main__":
    query = "machine learning"
    num_articles = 10
    articles = fetch_wikipedia_articles(query, num_articles)

    # Φόρτωση υπαρχόντων δεδομένων
    try:
        existing_data = pd.read_csv("data/wikipedia_articles.csv")
    except FileNotFoundError:
        existing_data = pd.DataFrame(columns=["title", "content", "url"])

    # Συνδυασμός δεδομένων
    df = pd.concat([existing_data, pd.DataFrame(articles)]).drop_duplicates(subset=["title"])
    df.to_csv("data/wikipedia_articles.csv", index=False)
    print(f"Αποθηκεύτηκαν {len(articles)} άρθρα στο data/wikipedia_articles.csv")
