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
from connect import *

def jongang_api(page):

    jongang_data = pd.DataFrame()

    url = jongurl.format( page)
    result = requests.get(url).text
    soup = bs(result,'lxml-xml')
    get = soup.find_all("e")

    for post in get:
      TITLE = post.find("TITLE").get_text()
      PUBLISHER = post.find("PUBLISHER").get_text()
      AUTHOR = post.find("AUTHOR").get_text()
      EDITION_STMT = post.find("EDITION_STMT").get_text()
      PRE_PRICE = post.find("PRE_PRICE").get_text()
      EA_ADD_CODE = post.find("EA_ADD_CODE").get_text()
      EBOOK_YN = post.find("EBOOK_YN").get_text()
      EA_ISBN = post.find("EA_ISBN").get_text()
      SUBJECT = post.find("SUBJECT").get_text()
      PUBLISH_PREDATE = post.find("PUBLISH_PREDATE").get_text()

      df = pd.DataFrame({"title":TITLE,
                           "author":AUTHOR,
                           "pub_year":PUBLISH_PREDATE,
                           "pub":PUBLISHER,
                           "isbn_add_code":EA_ADD_CODE,
                           "repub":EDITION_STMT,
                           "isbn":EA_ISBN,
                           "ebook":EBOOK_YN,
                           "price":PRE_PRICE,
                           "pub":PUBLISHER,
                           "SUBJECT":SUBJECT}, index = [0])
      
      jongang_data = pd.concat([jongang_data, df])
    
    db_process(jongang_data, 'jongang_isbn', 'project')
    
if __name__ == '__main__':
  
  jongang_api(i) # i = index값 1부터 2000년 2022 10월 44396

    
