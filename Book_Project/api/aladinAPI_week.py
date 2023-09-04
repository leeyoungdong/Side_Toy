import urllib
from urllib.request import Request, urlopen
import requests, bs4
from bs4 import BeautifulSoup as bs
import pandas as pd
import time
import datetime
import pymysql
import os
import re
from sqlalchemy import create_engine
import datetime
from lxml import html
from urllib.parse import urlencode, quote_plus, unquote
import datetime
from sqlite3 import OperationalError
from connect import *

#역으로 리스트 출력 
#input : 시작(큰 수), 끝(작은 수) 
def make_list_desc(start, end):
    temp = []
    for i in range(start, end-1, -1):
        temp.append(i)
    return temp

def make_zero_month(month):
    temp = str(month)
    smonth =temp.zfill(2)
    return smonth

#연도 주
def str_make_list_desc(start,end):
    temp = []
    for i in range(start, end-1, -1):
        tmp = str(i)
        temp.append(tmp)
    return temp

def wstr_make_list_desc(start,end):
    temp = []
    for i in range(start, end-1, -1):
        tmp = str(i)
        ztmp = tmp.zfill(2)
        temp.append(ztmp)
    return temp

def changeStr(num):
    strNum = str(num)
    return strNum

def Aladin_Week(year,month,week):

    aladin_week_data = pd.DataFrame()

    queryParams = '?' + urlencode(
        {
            quote_plus('ttbkey') : My_API_Key,
            quote_plus('QueryType') : 'Bestseller',
            quote_plus('MaxResults') : '100',
            quote_plus('start') : '1',
            quote_plus('SearchTarget') : 'Book',
            quote_plus('output') : 'xml',
            quote_plus('Version') : '20131101',
            quote_plus('Year') : year,
            quote_plus('Month') : month,
            quote_plus('Week') : week
        }
    )
    response = requests.get(Aladin_URL + queryParams).text.encode('utf-8')
    xmlobj = bs(response, 'lxml-xml')
    rows = xmlobj.findAll('item')
    
    week_date =""
    week_date = year+month+week
    
    for row in rows:
        try:
            title = row.find('title').text
            author = row.find('author').text
            pubDate = row.find('pubDate').text
            description = row.find('description').text
            isbn10 = row.find('isbn').text
            price =row.find('priceStandard').text
            publisher = row.find('publisher').text
            salesPoint = row.find('salesPoint').text
            rank = row.find('bestRank').text
            df =pd.DataFrame({'rank':rank,'title':title,'author':author,'publisher':publisher,'pubDate':pubDate,'description':description,'isbn10':isbn10,'price':price,'salesPoint':salesPoint} , index= [0])
            aladin_week_data = pd.concat([aladin_week_data,df],ignore_index= True)

        except AttributeError as e :
            print(e)
            pass

    aladin_week_data['wperiod']=week_date
    if len(aladin_week_data)==0:
        print('안들어가')
        pass

    else:
        db_process(aladin_week_data, 'alading_week', 'project')
        
    
if __name__ == "__main__":
    # year / month / week
    year_list = str_make_list_desc(2022, 2008)
    month_list = wstr_make_list_desc(12,1)
    week_list =str_make_list_desc(5,1)
    
    for year in year_list:
        if year =='2022':
            month = wstr_make_list_desc(11,1)
        for month in month_list:
            for week in week_list:
                print(year, month, week)
                Aladin_Week(year, month,week)
