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
from urllib.request import HTTPError
from book_connect import *

def hani_news(date):

    hani_data = pd.DataFrame()
  
    url = hani_url.format(date)
    response = urllib.request.urlopen(url)
    soup = bs(response,'html.parser')

    # 본문내용 크롤링
    try:
      category = soup.select_one("p.category > span").get_text()
      result = soup.select_one("h4 > span").get_text()
      pdate = soup.select_one('p.date-time > span:nth-child(1)').get_text()
      hani_data = pd.DataFrame({'category':category,'title':result,'pdate':pdate}, index=[0])
    
    except AttributeError as e: #중간에 영어뉴스가 존재하므로 예외처리
      print(e,'영어 뉴스 패스')
      pass

    db_process(hani_data, 'hani_news', 'project')


if __name__ == '__main__':
  hani_news(1) # 1 부터 1068411번째 뉴스까지 있음
