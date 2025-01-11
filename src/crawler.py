import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_wikipedia_articles(query, num_articles=10):
    base_url = "https://en.wikipedia.org/wiki/"
    search_url = f"https://en.wikipedia.org/w/index.php?search={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    links = [a['href'] for a in soup.select('a[href^="/wiki/"]') if ':' not in a['href']][:num_articles]

    articles = []
    for link in links:
        url = f"https://en.wikipedia.org{link}"
        article_response = requests.get(url)
        article_soup = BeautifulSoup(article_response.text, 'html.parser')

        title = article_soup.find('h1').text
        paragraphs = article_soup.find_all('p')
        content = ' '.join([p.text for p in paragraphs])

        articles.append({"title": title, "content": content, "url": url})

    return articles

if __name__ == "__main__":
    query = "machine learning"
    num_articles = 10
    articles = fetch_wikipedia_articles(query, num_articles)
    df = pd.DataFrame(articles)
    df.to_csv("data/wikipedia_articles.csv", index=False)
    print(f"Αποθηκεύτηκαν {len(articles)} άρθρα στο data/wikipedia_articles.csv")
