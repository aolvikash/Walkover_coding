import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bisect import bisect_left
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import time

def get_bbc_news(url):
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('li', class_='media-list__item')
    len(coverpage_news)
    number_of_articles = 5

    # Empty lists for content, links, titles and tag
    news_contents = []
    list_links = []
    list_titles = []
    list_tag = []

    for n in np.arange(0, number_of_articles):
        # Getting the link of the article
        link = coverpage_news[n].find('a')['href']
        list_links.append(link.strip())

        # Getting the title
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title.strip())

        # Getting the News Tag
        tag = coverpage_news[n].find('a', class_='media__tag').get_text()
        list_tag.append(tag.strip())

        if url in link:
            article = requests.get(link)
        else:
            article = requests.get(url + link)

        # Reading the content (it is divided in paragraphs)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')

        if soup_article.find_all('article', class_='article__body'):
            body = soup_article.find_all('article', class_='article__body')
            x = body[0].find_all('div')
        elif soup_article.find_all('div', class_='body-content'):
            body = soup_article.find_all('div', class_='body-content')
            x = body[0].find_all('div')
        elif soup_article.find_all('div', class_='lx-c-sticky__item'):
            body = soup_article.find_all('div', class_='lx-c-sticky__item')
            x = body[0].find_all('ol')
        elif soup_article.find_all('div', class_='vxp-media__summary'):
            body = soup_article.find_all('div', class_='vxp-media__summary')
            x = body[0].find_all('p')
        else:
            body = soup_article.find_all('div', class_='story-body__inner')
            x = body[0].find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
        news_contents.append(final_article.strip())

    #  df_features
    df_features = pd.DataFrame(
        {'Content': news_contents}
    )

    # df_show_info
    df_show_info = pd.DataFrame(
        {
            'Newspaper': 'The BBC',
            'News Tag': list_tag,
            'Article Title': list_titles,
            'Article Link': list_links,
            'Content': news_contents
         }
    )

    return (df_features, df_show_info)

def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = None
    display.max_rows = None
    display.max_colwidth = None
    display.width = None

def write_news_dataFrame(df,parquetFilePath):
    table_from_pandas = pa.Table.from_pandas(df)
    pq.write_table(table_from_pandas, parquetFilePath)

def read_news_from_parquet(parquetFilePath):
    df = pq.read_table(source=parquetFilePath).to_pandas()
    return df

def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i:
        return (i - 1)
    else:
        return -1

def get_nouns(text, chunk_func=ne_chunk):
    chunked = chunk_func(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

start = time.time()
# url definition
url = "https://www.bbc.com/"
parquetFilePath = "parquet/bbc_news_pandas.parquet"
news_content, news_df  = get_bbc_news(url)
set_pandas_display_options()
write_news_dataFrame(news_df,parquetFilePath)
parDF = read_news_from_parquet(parquetFilePath)
dotLine = "\n------------\n"
print (dotLine,"Top news",dotLine,parDF,dotLine)
noun_chunk = parDF['Content'].apply(lambda sent: get_nouns((sent)))
print (dotLine,"All the nouns",dotLine,noun_chunk,dotLine)
print (dotLine,"Binary Search for all the nouns",dotLine)
for i in range(len(noun_chunk)):
    for j in range(len(noun_chunk[i])):
        res = BinarySearch(parDF['Content'], noun_chunk[i][j])
        if res == -1:
            print(noun_chunk[i][j], "is absent")
        else:
            print("Last occurrence of", noun_chunk[i][j], "is present at", res)
    print()
end =time.time()
te = end-start
print("The time elapsed is %f seconds" %(te))
