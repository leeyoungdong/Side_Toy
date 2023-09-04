import time
import urllib
import json
import requests
from urllib.parse import urlencode, quote_plus, unquote
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus
from datetime import date
import datetime
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine

jongang_lib_url = 'https://www.nl.go.kr/NL/search/openApi/search.do?key={}&apiType=xml&detailSearch=true&category={}&sort=&pageNum={}&pageSize=100'

jongurl = 'https://www.nl.go.kr/seoji/SearchApi.do?cert_key=d8b083a071d52098f7ebc90f64ae6cb9bc66a6c47f0a6d10c4fd1f0bd5907c5c&result_style=xml&page_no={}&page_size=100'

Aladin_URL = 'https://www.aladin.co.kr/ttb/api/ItemList.aspx'

My_API_Key = unquote('ttbokok72722206001')

joongang_lib_key = 'd8b083a071d52098f7ebc90f64ae6cb9bc66a6c47f0a6d10c4fd1f0bd5907c5c'

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
