import requests
import pandas as pd

API_URL = "https://newsapi.org/v2/everything"
API_KEY = "a161ee259a74486db2f4ee1c399b0086"

def fetch_news(query, page_size=10, sort_by='popularity'):
    response = requests.get(API_URL, params={
        'q': query,
        'apiKey': API_KEY,
        'language': 'en',
        'pageSize': page_size,
        'sortBy': sort_by
    })

    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if articles:
            latest_news = pd.DataFrame(articles)
            latest_news['label'] = 0  
            latest_news['content'] = latest_news['title'] + " " + latest_news['description'].fillna('')
            return latest_news[['content', 'label']]
        else:
            return pd.DataFrame()
    else:
        print(f"Failed to fetch data, status code: {response.status_code}")
        return pd.DataFrame()