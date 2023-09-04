import time
import urllib
import json
import requests
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
from connect import *

def jongang_api(category, page):

    jongang_data = pd.DataFrame()

    word = urllib.parse.quote(category)
    url = jongang_lib_url.format(joongang_lib_key, word, page)
    result = requests.get(url)
    soup = bs(result.text,'lxml-xml')
    get = soup.find_all("item")

    for lib in get:
        title = lib.find("title_info").get_text()
        author = lib.find("author_info").get_text()
        pub = lib.find("pub_info").get_text()
        pub_year = lib.find('pub_year_info').get_text()
        type_name = lib.find('kdc_name_1s').get_text()

        df = pd.DataFrame({"title":title,
                           "author":author,
                           "pub_year":pub_year,
                           "pub":pub,
                           "type_name":type_name}, index = [0])

        jongang_data = pd.concat([jongang_data, df])

    db_process(jongang_data, 'jongang_lib_data','project')

if __name__ == '__main__':

    jongang_api('도서', i) # i = 1부터 48392까지의 484만개의 도서 데이터
