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


yes24_week_url = 'http://www.yes24.com/24/category/bestseller?CategoryNumber=001&sumgb=08&year={}&month={}&week={}&day=1&PageNumber={}' # 주간 week 5개있음음
yes24_day_url = 'http://www.yes24.com/24/category/bestseller?CategoryNumber=001&sumgb=07&year={}&month={}&day={}&PageNumber={}' # 일간
yes24_year_url = 'http://www.yes24.com/24/category/bestseller?categorynumber=001&sumgb=09&year={}&month={}&pagenumber={}' # 월간

donga_URL = 'https://www.donga.com/news/Pdf?ymd={}'

chosun_URL = 'https://archive.chosun.com/pdf/i_service/index_new_s.jsp?Y={}&M={}&D={}'

khan_URL = 'https://www.khan.co.kr/sitemap.html?year={}&month={}&day={}'

joongang_URL = 'https://www.joongang.co.kr/sitemap/index/{}/{}/{}'

hani_url= 'https://www.hani.co.kr/arti/culture/culture_general/{}.html'
# 1068411 인덱스로 카운팅

def db_process(df, table, database):

    con = pymysql.connect(host='localhost',
                            port=3306,
                            user='root',
                            password='lgg032800',
                            db=f'{database}',
                            charset='utf8')

    engine = create_engine(f'mysql+pymysql://root:lgg032800@localhost/{database}')
    df.to_sql(f'{table}',if_exists = 'append', con = engine)
    con.commit()
