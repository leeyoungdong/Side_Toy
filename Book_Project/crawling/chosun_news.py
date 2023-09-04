import time
import urllib
import json
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
import datetime
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from urllib.request import HTTPError
from book_connect import *

def chosun_crawl(year, month, day):

    chosun_data = pd.DataFrame()
    url = chosun_URL.format(year, month, day)
    response = urllib.request.urlopen(url)
    soup = bs(response,'html.parser')
    context_results = soup.select("div.ss_arti_tit")
    # 본문내용 크롤링
    for get in context_results: 
        context = get.select_one('div.ss_list').text
        con = context.split('\n')
        df = pd.DataFrame(con)
        chosun_data = chosun_data.append(df, ignore_index = True)

    news_date = datetime.date(year, month, day)
    
    day_results = soup.select('#LeftContent')
    for got in day_results:
        date = got.select('div.ss_txt')
    # 간단한 데이터 프레임 적재전 transform
    chosun_data.replace('', np.nan, inplace=True)
    chosun_data = chosun_data.dropna()
    chosun_data = chosun_data.reset_index()
    chosun_data = chosun_data.drop(columns='index')
    chosun_data['date'] = news_date

    db_process(chosun_data, 'chosun_news', 'project')

if __name__ == '__main__':
    chosun_crawl('year','month','day') # 돌리기전 try except 문으로 주말 뉴스가없어 에러가 뜨는사항에 대하여 예외처리 해줘야함