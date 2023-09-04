import pandas as pd
import re
import datetime
from stopword import joongang_stoplist, donga_stoplist, khan_stoplist, chosun_stoplist, hani_stoplist
import pymysql
from sqlalchemy import create_engine
from news_eda import *
from konlpy.tag import Okt
from collections import Counter
import re
import konlpy
from konlpy.tag import Okt
from konlpy.utils import pprint
from collections import Counter

"""
date 칼럼 format
조 2000-20-20 date
중 2020.20.20 00:00 date
동 20002020 date
경 20202020 date
한 등록 :2020-20-20 00: pdage
"""

# 형태소 분리기
okt = Okt()

# 기존 데이터 프레임 참조를 위한 Connect 연결
def connet():
    con = pymysql.connect(host='localhost',
                        port=3306,
                        user='root',
                        password='lgg032800',
                        db='project3',
                        charset='utf8')
    return con

#데이터 프레임 DB로 적재
def df_to_db(df, table):
    con = pymysql.connect(host='localhost',
                        port=3306,
                        user='root',
                        password='lgg032800',
                        db='project2',
                        charset='utf8')

    engine = create_engine('mysql+pymysql://root:lgg032800@localhost/project2')
    df.to_sql(f'{table}',if_exists = 'append', con = engine)
    con.commit()
    # con.close()

# NULL값 삭제 함수
def dropNull(df):
    df = df.dropna(subset=['title'])
    return df

def listEmpty(x):
    if len(x)== 0:
        y =" "
        return y
    else :
        y =" ".join(x)
        return y

def get_nouns(x):
    nouns_tagger = Okt()
    nouns = nouns_tagger.nouns(x)
    #EDA\stopwords.txt
    with open('C:/Users/youngdong/Book_Project/eda/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    stopwords = [x.strip() for x in stopwords]

    # 한글자 키워드 제거합니다
    nouns = [noun for noun in nouns if len(noun)>1]

    #불용어를 제거합니다
    nouns = [noun for noun in nouns if noun not in stopwords]

def donga_new(donga):

    # df = donga.drop([donga.columns[0]], axis=1)
    donga.columns = ['index', '0','date']
    df = donga.drop_duplicates(['0'])
    df_result = pd.DataFrame()

    for i in donga_stoplist:
        result = df[df['0'].str.contains(i,na=False)]
        df_result = pd.concat([df_result,result])
    
    df = pd.merge(df,df_result, how='outer', indicator=True)
    df = df.query('_merge == "left_only"').drop(columns=['_merge'])
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
    df = df.rename(columns={'0':'title'})
    df['company'] = 'donga'
    df = df.drop('index', axis=1)

    return df

def joongang_new(df):

    df.columns = ['index','context','date']
    df['context'] = df['context'].str.strip("\n")
    df = df.drop_duplicates(['context'])
    
    df_result = pd.DataFrame()
    for i in joongang_stoplist:
        result = df[df['context'].str.contains(i)]
        df_result = pd.concat([df_result,result])

    df = pd.merge(df,df_result, how='outer', indicator=True)
    df = df.query('_merge == "left_only"').drop(columns=['_merge'])
    df['date']= df['date'].str.slice(0,10)
    df['date']= pd.to_datetime(df['date'])
    df = df.rename(columns={'context':'title'})
    df['company'] = 'joongang'
    df = df.drop([df.columns[0]], axis=1)

    return df

def hani_new(hani):

    # df = hani.drop([hani.columns[0]], axis=1)
    hani.columns = ['index','category','title','pdate']
    df = hani[['title','pdate','category']]
    df = df.drop_duplicates(['title'])
    
    df_result = pd.DataFrame()
    for i in khan_stoplist:
        result = df[df['title'].str.contains(i,na=False)]
        df_result = pd.concat([df_result,result])

    df = pd.merge(df,df_result, how='outer', indicator=True)
    df = df.query('_merge == "left_only"').drop(columns=['_merge'])
    df['pdate'] = df['pdate'].str.slice(4,14)
    df = df.rename(columns={'pdate':'date'})
    df['company'] = 'hani'
    df = df[['title','date','company','category']]

    return df



def chosun_new(chosun):

    # df = chosun.drop([chosun.columns[0]], axis = 1)
    chosun.columns = ['index','0','date']
    df = chosun.rename(columns={'0':'title'})

    df = df.drop_duplicates(['title'])
    
    df_result = pd.DataFrame()

    for i in chosun_stoplist:
        result = df[df['title'].str.contains(i)]
        df_result = pd.concat([df_result,result])

    df = pd.merge(df, df_result, how='outer', indicator=True)
    df = df.query('_merge == "left_only"').drop(columns=['_merge'])
    df['company'] = 'chosun'
    df = df.drop('index', axis=1)

    return df

def khan_new(df):

    df.columns = ['index','context','date']
    df = df.drop_duplicates(['context'])
    
    df_result = pd.DataFrame()
    
    for i in hani_stoplist:
        result = df[df['context'].str.contains(i,na=False)]
        df_result = pd.concat([df_result,result])
    
    df = pd.merge(df,df_result, how='outer', indicator=True)
    df = df.query('_merge == "left_only"').drop(columns=['_merge'])
    df['date'] = df['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
    df = df.rename(columns={'context':'title'})
    df['company'] = 'khan'
    df = df.drop([df.columns[0]], axis=1)

    return df

def df_sep(df, table):
    # key값 생성을 위해 기존 DB를 참조해 인덱스 값을 확인함
    con = connet()
    cursor = con.cursor()
    a = f"""select count(*) from project2.news_table_daily;"""
    b = cursor.execute(a)
    c = cursor.fetchone()[0]
    # Base 기본 데이터 프레임 생성
    base = pd.DataFrame(df)
    base = base.reset_index()
    # 칼럼 분리를 위해 각각의 데이터 프레임 생성
    news_table = pd.DataFrame()
    CONTEXT = pd.DataFrame()
    DATE =pd.DataFrame()
    # 데이터 칼럼 분리 및 키값 더하기
    # base = base.apply(lambda x: index_key(x['company'],c), axis =1)
    #base = base.reset_index()
    base['index'] = base.index + c
    base['index'] = base['index'].astype(str)
    #base['index'] = base['company'].apply(lambda x: index_key(x['company']))
    base['date'] = pd.to_datetime(base['date'])
    base['year'] = base['date'].dt.year
    base['month']= base['date'].dt.month
    base['day']= base['date'].dt.day
    DATE = base[['year','month', 'day']]
    
    #item키 생성 방식
    base['cc']= base['company'].str[:1]
    base['index'] = base['index']+ base['cc']

    #이 item 키를 가지고 news_table과 DATE를 만듬
    DATE['index'] = base['index']
    news_table['context']= base['title']
    news_table['index']= base['index']
    base['title_nouns']= base['title'].apply(lambda x: get_nouns(x))
    CONTEXT['context']=base['title']
    CONTEXT['news_publisher']= base['company']
    #CONTEXT['word_one'] = base['title_nouns']
    base['title_noun_text']=  base['title_nouns'].apply(lambda x : listEmpty(x))
    CONTEXT['word_one']= base['title_noun_text']
    if(table == 'hani_news'):
        CONTEXT['category']= base['category']
    else:
        CONTEXT['category']= " "

    # 각 데이터 프레임을 테이블로 DB에 적재
    df_to_db(CONTEXT, 'context_daily')
    df_to_db(DATE, 'date_daily')
    df_to_db(news_table, 'news_table_daily')

if __name__ == "__main__":
    # 뉴스 입력값 구조
    # df = dropNull(donga_new(donga))
    # df_sep(df, 'donga_news') 
    print('c')