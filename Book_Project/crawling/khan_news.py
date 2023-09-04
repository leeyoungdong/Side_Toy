import urllib
from urllib.request import Request, urlopen
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import datetime
import pymysql
import os
import re
from sqlalchemy import create_engine
import datetime
from book_connect import *

def khan(year, month, day):
    
    khan_data = pd.DataFrame()

    headers={'User-Agent': 'Mozilla/5.0'}
    url = khan_URL.format(year, month, day)
    request = Request(url, headers=headers)
    html = urlopen(request).read()
    soup = bs(html, 'html.parser')
    result = soup.select('ul.daynews_list > li')
    # 본문내용 크롤링
    for posts in result:
        context = posts.select_one('[title]').text
        df = pd.DataFrame({'context':context}, index = [0])
        khan_data = khan_data.append(df, ignore_index = True)

    news_date = year+month+day
    khan_data['date'] = news_date
    db_process(khan_data, 'khan_news', 'project')
    
if __name__ == '__main__':
    khan('year','month','day')