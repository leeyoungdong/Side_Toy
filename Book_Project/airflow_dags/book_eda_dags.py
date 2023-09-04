import pandas as pd
import numpy as np
import re
import datetime
import pymysql
from sqlalchemy import create_engine
from news_main_temp import df_sep, dropNull
from news_eda import *


############################################## PROJECT. () 바꾸기!!
con = pymysql.connect(host='localhost',
                        port=3306,
                        user='root',
                        password='lgg032800',
                        db='project2',
                        charset='utf8')

engine = create_engine('mysql+pymysql://root:lgg032800@localhost/project1')

cursor = con.cursor()
# project.  ---- db 이름

# hani 만 pdade
def joongang_db(year, month, day):

    sql  =  f"""select * from project.joongang_news
                where date like ('%{year}.{month}.{day}%');"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result = result.drop(0, axis= 1)

    df = dropNull(joongang_new(result))
    df_sep(df, 'result')

def chosun_db(year, month, day):

    sql  =  f"""select * from project.chosun_news
                where date like '%{year}-{month}-{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result = result.drop(0, axis= 1)

    df = dropNull(chosun_new(result))
    df_sep(df, 'result')

def hani_db(year, month, day):

    sql  =  f"""select * from project.hani_news
                where pdate like '%{year}-{month}-{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result = result.drop(0, axis= 1)

    df = dropNull(hani_new(result))
    df_sep(df, 'result')

def khan_db(year, month, day):

    sql  =  f"""select * from project.khan_news
                where date like '%{year}{month}{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result = result.drop(0, axis= 1)
 
    df = dropNull(khan_new(result))
    df_sep(df, 'result')

def donga_db(year, month, day):

    sql  =  f"""select * from project.donga_news
                where date like '%{year}{month}{day}%';"""
    cursor.execute(sql)
    result = cursor.fetchall()
    result = pd.DataFrame(result)
    result = result.drop(0, axis= 1)

    df = dropNull(donga_new(result))
    print(df)
    df_sep(df, 'result')

if __name__ == "__main__"
    print('try')