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

def yes24_day(year, month, day, page):
    
    yes24_day_data = pd.DataFrame()

    url = yes24_day_url.format(year, month, day, page)
    response = urllib.request.urlopen(url)
    soup = bs(response, 'html.parser')
    result = soup.select('td.goodsTxtInfo')
    day_date = str(year)+ str(month) + str(day)
    # 본문내용 크롤링
    for i, post in enumerate(result):
      try:
        rank =  soup.select_one(f'#goods{i+1+20*(page-1)}').text
        review = post.select_one('p.review > a').text
        context = post.select_one('p:nth-of-type(1)').text
        auther = post.select_one('div').text
        df = pd.DataFrame({'rank':rank,'context':context,'review':review,'auther':auther}, index = [0])
        yes24_day_data = pd.concat([yes24_day_data,df])

      except AttributeError as e: # 주말 및 공휴일은 뉴스가 없으므로 error발생
        print(e)
        print(i)
        pass

    yes24_day_data['date'] = day_date
    db_process(yes24_day_data, 'yes24_day', 'project')
     

def yes24_week(year, month, week, page):
    
    yes24_week_data = pd.DataFrame()

    url = yes24_week_url.format(year, month, week, page)
    response = urllib.request.urlopen(url)
    soup = bs(response, 'html.parser')
    result = soup.select('td.goodsTxtInfo')
    week_date = str(year)+ str(month) + str(week)
    # 본문내용 크롤링
    for i, post in enumerate(result):
      try:
        rank =  soup.select_one(f'#goods{i+1+20*(page-1)}').text
        review = post.select_one('p.review > a').text
        context = post.select_one('p:nth-of-type(1)').text
        auther = post.select_one('div').text
        df = pd.DataFrame({'rank':rank,'context':context,'review':review,'auther':auther}, index = [0])
        yes24_week_data = pd.concat([yes24_week_data,df])

      except AttributeError as e: # 주말 및 공휴일은 뉴스가 없으므로 error발생
        print(e)        
        print(i)
        pass
      
    yes24_week_data['date'] = week_date
    db_process(yes24_week_data, 'yes24_week', 'project')
    

def yes24_year(year, month, page):
    
    yes24_year_data = pd.DataFrame()

    url = yes24_year_url.format(year, month, page)
    response = urllib.request.urlopen(url)
    soup = bs(response, 'html.parser')
    result = soup.select('td.goodsTxtInfo')
    year_date = str(year)+ str(month)
    # 본문내용 크롤링
    for i, post in enumerate(result):
      try:
        rank =  soup.select_one(f'#goods{i+1+20*(page-1)}').text
        review = post.select_one('p.review > a').text
        context = post.select_one('p:nth-of-type(1)').text
        auther = post.select_one('div').text
        df = pd.DataFrame({'rank':rank,'context':context,'review':review,'auther':auther}, index = [0])
        yes24_year_data = pd.concat([yes24_year_data,df])

      except AttributeError as e: # 주말 및 공휴일은 뉴스가 없으므로 error발생
        print(e)
        print(i)
        pass

    yes24_year_data['date'] = year_date
    db_process(yes24_year_data, 'yes24_year', 'project')
    

if __name__ == "__main__":
  for i in range(1,10):
    a = datetime.datetime(2008, 1, 1) + datetime.timedelta(days= i - 1)
    yes24_year(a.strftime("%Y"),a.strftime("%m"),'page') # 입력값 구조
    yes24_day(a.strftime("%Y"),a.strftime("%m"),a.strftime("%d"),'page' ) # 입력값 구조
    yes24_week(a.strftime("%Y"),a.strftime("%m"),'week','page') # 입력값 구조