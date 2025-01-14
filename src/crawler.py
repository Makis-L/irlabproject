import requests
from bs4 import BeautifulSoup
import json

def fetch_wikipedia_articles(query, num_articles=15):
    # Βασικά URLs
    search_url = f"https://en.wikipedia.org/w/index.php?search={query}"
    
    # Αναζήτηση στο Wikipedia
    response = requests.get(search_url)
    if response.status_code != 200:
        print("Σφάλμα κατά την ανάκτηση της σελίδας αναζήτησης.")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Εύρεση συνδέσμων άρθρων
    links = [a['href'] for a in soup.select('a[href^="/wiki/"]') if ':' not in a['href']][:num_articles]

    articles = []
    for link in links:
        url = f"https://en.wikipedia.org{link}"
        article_response = requests.get(url)
        if article_response.status_code != 200:
            print(f"Σφάλμα κατά την ανάκτηση του άρθρου: {url}")
            continue
        
        article_soup = BeautifulSoup(article_response.text, 'html.parser')

        # Εξαγωγή τίτλου και περιεχομένου
        title = article_soup.find('h1').text
        paragraphs = article_soup.find_all('p')
        content = ' '.join([p.text.strip() for p in paragraphs])

        # Προσθήκη του άρθρου
        articles.append({
            "title": title,
            "url": url,
            "content": content
        })

    return articles

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Τα άρθρα αποθηκεύτηκαν στο αρχείο: {filename}")

if __name__ == "__main__":
    query = "Social Media Entertainment"  # Ερώτημα αναζήτησης
    num_articles = 15  # Αριθμός άρθρων
    articles = fetch_wikipedia_articles(query, num_articles)

    if articles:
        save_to_json(articles, "wikipedia_articles.json")
