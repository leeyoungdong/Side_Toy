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

def donga(date):

    donga_data = pd.DataFrame()
    url = donga_URL.format(date)
    response = urllib.request.urlopen(url)
    soup = bs(response,'html.parser')
    result = soup.select('div.section_txt')

    # 본문내용 크롤링
    for post in result:
        try:
            context = post.select_one('ul.desc_list').text
            con = context.split('\r')
            df = pd.DataFrame(con)
            donga_data = donga_data.append(df, ignore_index = True)

        except AttributeError as e: # 주말 및 공휴일은 뉴스가 없으므로 error발생
            print(e)
            pass
    
    donga_data['date'] = str(date)
    db_process(donga_data, 'donga_news', 'project')


if __name__ == '__main__':
    donga('yearmonthday') # 입력값 구조 
