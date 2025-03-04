import json
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Base URL for crawling
BASE_URL = "https://jiopay.com/business"
VISITED_URLS = set()
DATA = []

def is_internal_link(link, base_domain):
    """ Check if a link is internal """
    parsed_link = urlparse(link)
    return base_domain in parsed_link.netloc or parsed_link.netloc == ''

def get_child_links(soup, base_url):
    """ Extracts all internal links from a page """
    links = set()
    for a_tag in soup.find_all('a', href=True):
        url = urljoin(base_url, a_tag['href'])
        if is_internal_link(url, urlparse(BASE_URL).netloc):
            links.add(url)
    return links

def extract_content(soup):
    """ Extracts relevant content from a BeautifulSoup-parsed page """
    title = soup.title.string if soup.title else 'No Title'
    paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
    content = " ".join(paragraphs)
    return title, content

def scrape_page(page, url):
    """ Navigates to a URL and extracts data """
    page.goto(url)
    html = page.content()
    soup = BeautifulSoup(html, 'html.parser')
    
    title, content = extract_content(soup)
    DATA.append({
        'url': url,
        'title': title,
        'content': content
    })
    print(f"Scraped: {url}")

    child_links = get_child_links(soup, url)
    return child_links

def recursive_scrape(page, url):
    """ Recursively scrape all internal links """
    if url not in VISITED_URLS:
        VISITED_URLS.add(url)
        child_links = scrape_page(page, url)
        for link in child_links:
            recursive_scrape(page, link)

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Start recursive scraping from the base URL
        recursive_scrape(page, BASE_URL)
        
        browser.close()
        
        # Save the scraped data to a JSON file
        with open('jiopay_data.json', 'w') as f:
            json.dump(DATA, f, indent=4)
        print("Scraping completed. Data saved to jiopay_data.json")

if __name__ == "__main__":
    main()
