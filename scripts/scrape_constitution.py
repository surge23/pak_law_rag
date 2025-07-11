import requests
from bs4 import BeautifulSoup
import json
import re
import time

BASE_URL = "http://pakistani.org/pakistan/constitution/"
INDEX_URL = BASE_URL + "index.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def safe_request(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return requests.get(url, headers=HEADERS, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    raise Exception(f"❌ Failed after {retries} attempts: {url}")

def get_article_links():
    response = safe_request(INDEX_URL)
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a", href=True)

    article_links = []
    for link in links:
        href = link['href']
        if href.startswith("part") and href.endswith(".html"):
            full_url = BASE_URL + href
            title = link.get_text(strip=True) or href.replace(".html", "")
            article_links.append((title, full_url))

    return article_links

def extract_article_text(url):
    response = safe_request(url)
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all(["p", "h3", "h4"])
    combined = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
    return re.sub(r'\s+', ' ', combined).strip()

def scrape_constitution():
    print("🔍 Scraping index...")
    links = get_article_links()
    print(f"🔗 Found {len(links)} valid article pages.\n")

    articles = []
    for i, (title, url) in enumerate(links):
        print(f"[{i + 1}/{len(links)}] Fetching {title} ...")
        try:
            text = extract_article_text(url)
            articles.append({
                "title": title,
                "url": url,
                "text": text
            })
            time.sleep(1)  # To avoid getting blocked
        except Exception as e:
            print(f"❌ Error: {e}")

    with open("../data/constitution_clean.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Saved {len(articles)} articles to constitution_clean.json.")

if __name__ == "__main__":
    scrape_constitution()
