import time
import urllib
import json
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
from datetime import date
import datetime
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from urllib.request import HTTPError
from book_connect import *

def joongang_crawl(year, month, day):


    joongang_data = pd.DataFrame()
    url = joongang_URL.format(year, month, day)
    response = urllib.request.urlopen(url)
    soup = bs(response,'html.parser')
    result = soup.select("li.card")

    # 본문내용 크롤링
    for post in result:
        context = post.select_one("h2.headline").text

        try:
            date = post.select_one("p.date")
        except AttributeError as e: # 주말 및 공휴일은 뉴스가 없으므로 error발생
            print(e)
            pass
        df = pd.DataFrame({'context':context,'date':date}, index = [0])
        joongang_data = joongang_data.append(df, ignore_index = True)
    
    joongang_data = joongang_data[5:] # 앞쪽에 공백내용이 같이 크롤링됨
    db_process(joongang_data, 'joongang_news', 'project')



if __name__ == '__main__':
    joongang_crawl('yearmonthday') # 11월 현재 8360일정도 크롤링 해야함

