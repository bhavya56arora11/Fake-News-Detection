from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def extract_claim_from_title(title):
    correction_keywords = [
        "falsely", "misleading", "unrelated", "false", "debunked", "wrong", "fact-check",
        "incorrect", "inaccurate", "no connection", "fact check", "truth"
    ]
    
    for keyword in correction_keywords:
        title = re.split(r'\b' + keyword + r'\b', title, flags=re.IGNORECASE)[0]
    
    return title.strip()

def scrape_fake_claims():
    news_data = []
    base_url = "https://factly.in/category/english/page/"

    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    driver = webdriver.Chrome(options=chrome_options)

    for page in range(1, 12):  
        url = f"{base_url}{page}/"
        print(f"Fetching data from: {url}")

        try:
            driver.get(url)

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)  

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            articles = soup.find_all('article')

            for article in articles:
                title_tag = article.find('h2', class_='post-title')
                title = title_tag.get_text(strip=True) if title_tag else 'No Title'
                
                claim_tag = article.find('div', class_='excerpt')
                claim = claim_tag.get_text(strip=True) if claim_tag else ''
                
                false_claim = extract_claim_from_title(title)
                
                content = false_claim + " " + claim
                if content.strip():  
                    news_data.append({"content": content.strip(), "label": 1})

            time.sleep(4)  
        except Exception as e:
            print(f"Error fetching data from {url}: {e}")

    driver.quit()
    return pd.DataFrame(news_data)

if __name__ == "__main__":
    fake_news = scrape_fake_claims()
    if not fake_news.empty:
        print(fake_news.head())
    else:
        print("No data extracted.")