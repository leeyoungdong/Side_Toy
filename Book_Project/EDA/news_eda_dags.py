import pandas as pd
import numpy as np
import re
import datetime
import pymysql
from sqlalchemy import create_engine
from news_eda import *


# airflow 작업시 host(), user, password, engine 바꾸기!!
con = pymysql.connect(host='localhost',
                        port=3306,
                        user='root',
                        password='lgg032800',
                        db='project3',
                        charset='utf8')

engine = create_engine('mysql+pymysql://root:lgg032800@localhost/project3')

cursor = con.cursor()

# hani만 날짜관련 칼럼이름이 pdade
# 중앙 뉴스 DB행
def joongang_db(year, month, day):

    sql  =  f"""select * from project3.joongang_news_daily
                where date like ('%{year}.{month}.{day}%');"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.columns = ['index','context','date']
    # result = result.drop(0, axis= 1)

    df = dropNull(joongang_new(result))
    df_sep(df, 'result')

# 조선 뉴스 DB행
def chosun_db(year, month, day):

    sql  =  f"""select * from project3.chosun_news_daily
                where date like '%{year}-{month}-{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.columns = ['index','0','date']
    # result = result.drop(0, axis= 1)

    df = dropNull(chosun_new(result))
    df_sep(df, 'result')

# 한겨례 뉴스 DB행
def hani_db(year, month, day):

    sql  =  f"""select * from project3.hani_news_daily
                where pdate like '%{year}-{month}-{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.columns = ['index','category','title','pdate']
    # result = result.drop(0, axis= 1)

    df = dropNull(hani_new(result))
    df_sep(df, 'result')

# 경향 뉴스 DB행
def khan_db(year, month, day):

    sql  =  f"""select * from project3.khan_news_daily
                where date like '%{year}{month}{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.columns = ['index','context','date']
    # result = result.drop(0, axis= 1)
 
    df = dropNull(khan_new(result))
    df_sep(df, 'result')

# 동아 뉴스 DB행
def donga_db(year, month, day):

    sql  =  f"""select * from project3.donga_news_daily
                where date like '%{year}{month}{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result.columns = ['index', '0','date']
    # result = result.drop(0, axis= 1)

    df = dropNull(donga_new(result))
    print(df)
    df_sep(df, 'result')

if __name__ == "__main__":

    a = datetime.datetime.now()
    khan_db(a.strftime("%Y"),a.strftime("%m"),a.strftime("%d"))
    donga_db(a.strftime("%Y"),a.strftime("%m"),a.strftime("%d"))
    chosun_db(a.strftime("%Y"),a.strftime("%m"),a.strftime("%d"))
    joongang_db(a.strftime("%Y"),a.strftime("%m"),a.strftime("%d"))