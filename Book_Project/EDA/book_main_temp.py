import pandas as pd
import numpy as np
import re
import pymysql
from sqlalchemy import create_engine
from book_eda import *
from book_main import *

# 교보 월간
def kb_m(df):

    period, book_table, information, reputation, buyc = base_df_create()

    kb_m = pd.DataFrame(df)
    kb_m = pd.DataFrame(kb_m)


    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    # kb_m = kb_m.drop('0', axis = 1)
    kb_m = kb_m.reset_index()
    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'k'+'m'
    kb_m['저자'] = dup_count(kb_m, '저자')
    kb_m['저자'] = kb_m['저자'] .astype('str')
    kb_m['저자']  = kb_m['저자']  + 'k'+'m'
    kb_m['제목'] = kb_m['제목'].str.upper()
    kb_m['제목'] = dup_count(kb_m, '제목')
    kb_m['제목'] = kb_m['제목'].astype('str')
    kb_m['제목'] = kb_m['제목']  + kb_m['index']
    kb_m['기간'] = kb_m['기간'].astype('str')
    kb_m['기간'] = kb_m['기간'] + 'k'
    kb_m['기간'] = dup_count(kb_m, '기간')

    period['date'] = kb_m['기간'] 
    period['pub_date'] = kb_m['출판연도']
    period['year'] = kb_m['기간'].str.slice(0,4)
    period['month'] = kb_m['기간'].str.slice(4,6)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['기간']
    book_table['title'] = kb_m['제목']
    book_table['author'] = kb_m['저자']

    information['title'] = kb_m['제목']
    information['category'] = kb_m['카테고리']
    information['publisher'] = kb_m['출판사']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['순위']
    reputation['review_num'] = kb_m['리뷰개수']
    reputation['review_rate'] = kb_m['평점']
    reputation['portal'] = 'kyobo'+'m'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']
    buyc['portal'] = dup_count(buyc, 'portal')
    buyc = buyc.drop('new1',axis=1)

    df_to_db(period, book_table, information, reputation, buyc)
    
# 교보 주간
def kb_w(df):

    kb_w = pd.DataFrame(df)
    kb_m = pd.DataFrame(kb_w)

    period, book_table, information, reputation, buyc = base_df_create()

    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    kb_m = kb_m.reset_index()
    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'k'+ 'w'
    kb_m['저자'] = dup_count(kb_m, '저자')
    kb_m['저자'] = kb_m['저자'] .astype('str')
    kb_m['저자']  = kb_m['저자']  + 'k' + 'w'
    kb_m['제목'] = kb_m['제목'].str.upper()
    kb_m['제목'] = dup_count(kb_m, '제목')
    kb_m['제목'] = kb_m['제목'].astype('str')
    kb_m['제목'] = kb_m['제목'] + kb_m['index']
    kb_m['기간'] = kb_m['기간'].astype('str')
    kb_m['기간'] = kb_m['기간'] + 'k'+ 'w'
    kb_m['기간'] = dup_count(kb_m, '기간')

    period['date'] = kb_m['기간'] 
    period['pub_date'] = kb_m['출판연도']
    period['year'] = kb_m['기간'].str.slice(0,4)
    period['month'] = kb_m['기간'].str.slice(4,6)
    period['week'] = kb_m['기간'].str.slice(6,7)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['기간']
    book_table['title'] = kb_m['제목']
    book_table['author'] = kb_m['저자']

    information['title'] = kb_m['제목']
    information['category'] = kb_m['카테고리']
    information['publisher'] = kb_m['출판사']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['순위']
    reputation['review_num'] = kb_m['리뷰개수']
    reputation['review_rate'] = kb_m['평점']
    reputation['portal'] = 'kyobo'+ 'w'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']
    buyc['portal'] = dup_count(buyc, 'portal')
    buyc = buyc.drop('new1',axis=1)

    df_to_db(period, book_table, information, reputation, buyc)

#교보 연간
def kb_y(df):

    kb_y= pd.DataFrame(df)
    kb_m = pd.DataFrame(kb_y)

    period, book_table, information, reputation, buyc = base_df_create()

    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    kb_m = kb_m.reset_index()
    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'k'+'y'
    kb_m['저자'] = dup_count(kb_m, '저자')
    kb_m['저자'] = kb_m['저자'] .astype('str')
    kb_m['저자']  = kb_m['저자']  + 'k'+'y'
    kb_m['제목'] = kb_m['제목'].str.upper()
    kb_m['제목'] = dup_count(kb_m, '제목')
    kb_m['제목'] = kb_m['제목'].astype('str')
    kb_m['제목'] = kb_m['제목']  + kb_m['index']
    kb_m['기간'] = kb_m['기간'].astype('str')
    kb_m['기간'] = kb_m['기간'] + 'k'+ 'y'
    kb_m['기간'] = dup_count(kb_m, '기간')

    period['date'] = kb_m['기간'] 
    period['pub_date'] = kb_m['출판연도']
    period['year'] = kb_m['기간'].str.slice(0,4)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['기간']
    book_table['title'] = kb_m['제목']
    book_table['author'] = kb_m['저자']

    information['title'] = kb_m['제목']
    information['category'] = kb_m['카테고리']
    information['publisher'] = kb_m['출판사']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['순위']
    reputation['review_num'] = kb_m['리뷰개수']
    reputation['review_rate'] = kb_m['평점']
    reputation['portal'] = 'kyobo' + 'y'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']
    buyc['portal'] = dup_count(buyc, 'portal')
    buyc = buyc.drop('new1',axis=1)

    df_to_db(period, book_table, information, reputation, buyc)

#yes24 일간
def yes_d(df):

    yes_d = pd.DataFrame(df)
    kb_m = pd.DataFrame(yes_d)

    period, book_table, information, reputation, buyc = base_df_create()

    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    # kb_m = kb_m.drop('0', axis = 1)
    kb_m = kb_m.reset_index()
    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'y'+'d'
    kb_m['auther'] = dup_count(kb_m, 'auther')
    kb_m['auther'] = kb_m['auther'] .astype('str')
    kb_m['auther']  = kb_m['auther'] + 'y'+'d'
    kb_m['context'] = kb_m['context'].str.upper()
    kb_m['context'] = dup_count(kb_m, 'context')
    kb_m['context'] = kb_m['context'].astype('str')
    kb_m['context'] = kb_m['context']  + kb_m['index']
    kb_m['r_date'] = kb_m['r_date'].astype('str')
    kb_m['r_date'] = kb_m['r_date']+'y' + 'd'
    kb_m['r_date'] = dup_count(kb_m, 'r_date')

    period['date'] = kb_m['r_date']
    period['pub_date'] = kb_m['publication']
    period['year'] = kb_m['r_date'].str.slice(0,4)
    period['month'] = kb_m['r_date'].str.slice(5,7)
    period['day'] = kb_m['r_date'].str.slice(8,10)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['r_date']
    book_table['title'] = kb_m['context']
    book_table['author'] = kb_m['auther']

    information['title'] = kb_m['context']
    information['publisher'] = kb_m['publisher']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['b_rank']
    reputation['review_num'] = kb_m['review']
    reputation['portal'] = 'yes' + 'd'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']

    df_to_db(period, book_table, information, reputation, buyc)

#yes24 월간
def yes_m(df):

    yes_m = pd.DataFrame(df)
    kb_m = pd.DataFrame(yes_m)

    period, book_table, information, reputation, buyc = base_df_create()

    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    # kb_m = kb_m.drop('0', axis = 1)
    kb_m = kb_m.reset_index()
    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'y'+'m'
    kb_m['auther'] = dup_count(kb_m, 'auther')
    kb_m['auther'] = kb_m['auther'] .astype('str')
    kb_m['auther']  = kb_m['auther'] + 'y'+'m'
    kb_m['context'] = kb_m['context'].str.upper()
    kb_m['context'] = dup_count(kb_m, 'context')
    kb_m['context'] = kb_m['context'].astype('str')
    kb_m['context'] = kb_m['context'] + kb_m['index']
    kb_m['r_date'] = kb_m['r_date'].astype('str')
    kb_m['r_date'] = kb_m['r_date']+'y' + 'm'
    kb_m['r_date'] = dup_count(kb_m, 'r_date')

    period['date'] = kb_m['r_date']
    period['pub_date'] = kb_m['publication']
    period['year'] = kb_m['r_date'].str.slice(0,4)
    period['month'] = kb_m['r_date'].str.slice(4,6)
    period['week'] = kb_m['r_date'].str.slice(6,7)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['r_date']
    book_table['title'] = kb_m['context']
    book_table['author'] = kb_m['auther']

    information['title'] = kb_m['context']
    information['publisher'] = kb_m['publisher']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['b_rank']
    reputation['review_num'] = kb_m['review']
    reputation['portal'] = 'yes' + 'm'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']

    df_to_db(period, book_table, information, reputation, buyc)

#yes24 월간
def yes_y(df):

    yes_y = pd.DataFrame(df)
    kb_m = pd.DataFrame(yes_y)

    period, book_table, information, reputation, buyc = base_df_create()

    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    # kb_m = kb_m.drop('0', axis = 1)
    kb_m = kb_m.reset_index()
    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'y'+'y'
    kb_m['auther'] = dup_count(kb_m, 'auther')
    kb_m['auther'] = kb_m['auther'] .astype('str')
    kb_m['auther']  = kb_m['auther'] + 'y'+'y'
    kb_m['context'] = kb_m['context'].str.upper()
    kb_m['context'] = dup_count(kb_m, 'context')
    kb_m['context'] = kb_m['context'].astype('str')
    kb_m['context'] = kb_m['context'] + kb_m['index']
    kb_m['r_date'] = kb_m['r_date'].astype('str')
    kb_m['r_date'] = kb_m['r_date']+'y' + 'y'
    kb_m['r_date'] = dup_count(kb_m, 'r_date')

    period['date'] = kb_m['r_date']
    period['pub_date'] = kb_m['publication']
    period['year'] = kb_m['r_date'].str.slice(0,4)
    period['month'] = kb_m['r_date'].str.slice(4,6)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['r_date']
    book_table['title'] = kb_m['context']
    book_table['author'] = kb_m['auther']

    information['title'] = kb_m['context']
    information['publisher'] = kb_m['publisher']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['b_rank']
    reputation['review_num'] = kb_m['review']
    reputation['portal'] = 'yes' + 'y'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']

    df_to_db(period, book_table, information, reputation, buyc)

# 인터파크 연간
def inter_y(df):

    inter_y = pd.DataFrame(df)
    kb_m = pd.DataFrame(inter_y)


    period, book_table, information, reputation, buyc = base_df_create()


    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    # kb_m = kb_m.drop('0', axis = 1)
    kb_m = kb_m.reset_index()

    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'i'+'y'
    kb_m['author'] = dup_count(kb_m, 'author')
    kb_m['author'] = kb_m['author'] .astype('str')
    kb_m['author']  = kb_m['author'] + 'i'+'y'
    kb_m['title'] = kb_m['title'].str.upper()
    kb_m['title'] = dup_count(kb_m, 'title')
    kb_m['title'] = kb_m['title'].astype('str')
    kb_m['title'] = kb_m['title'] + kb_m['index']
    kb_m['date'] = kb_m['date'].astype('str')
    kb_m['date'] = kb_m['date']+ 'i'+'y'
    kb_m['date'] = dup_count(kb_m, 'date')

    period['date'] = kb_m['date']
    period['year'] = kb_m['date'].str.slice(0,4)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['date']
    book_table['title'] = kb_m['title']
    book_table['author'] = kb_m['author']

    information['title'] = kb_m['title']
    information['category'] = kb_m['category']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['rank']
    reputation['review_rate'] = kb_m['review']
    reputation['portal'] = 'interpark'+'ｙ'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']
    buyc['accucnt'] = kb_m['accuCnt']
    buyc['aggrcnt'] = kb_m['aggrCnt']
    buyc['sales'] =  kb_m['구매력?']

    df_to_db(period, book_table, information, reputation, buyc)

#인터파크 월간
def inter_m(df):

    inter_w = pd.DataFrame(df)
    kb_m = pd.DataFrame(inter_w)

    period, book_table, information, reputation, buyc = base_df_create()

    kb_m = kb_m.reset_index()
    kb_m = kb_m.drop('index', axis = 1)
    # kb_m = kb_m.drop('0', axis = 1)
    kb_m = kb_m.reset_index()
    kb_m['index'] = kb_m['index'].astype('str')
    kb_m['index'] = kb_m['index'] + 'i'+'m'
    kb_m['author'] = dup_count(kb_m, 'author')
    kb_m['author'] = kb_m['author'] .astype('str')
    kb_m['author']  = kb_m['author'] + 'i'+'m'
    kb_m['title'] = kb_m['title'].str.upper()
    kb_m['title'] = dup_count(kb_m, 'title')
    kb_m['title'] = kb_m['title'].astype('str')
    kb_m['title'] = kb_m['title']  + kb_m['index']
    kb_m['date'] = kb_m['date'].astype('str')
    kb_m['date'] = kb_m['date']+ 'i'+'m'
    kb_m['date'] = dup_count(kb_m, 'date')

    period['date'] = kb_m['date']
    period['year'] = kb_m['date'].str.slice(0,4)
    period['month'] = kb_m['date'].str.slice(4,6)

    book_table['itemkey'] = kb_m['index']
    book_table['date'] = kb_m['date']
    book_table['title'] = kb_m['title']
    book_table['author'] = kb_m['author']

    information['title'] = kb_m['title']
    information['category'] = kb_m['category']

    reputation['itemkey'] = kb_m['index']
    reputation['rank'] = kb_m['rank']
    reputation['review_rate'] = kb_m['review']
    reputation['portal'] = 'interpark'+'m'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']
    buyc['accucnt'] = kb_m['accuCnt']
    buyc['aggrcnt'] = kb_m['aggrCnt']
    buyc['sales'] =  kb_m['구매력?']

    df_to_db(period, book_table, information, reputation, buyc)

# 알라딘 주간
def aladin(df):

    aladin = pd.DataFrame(df)
    aladin.columns = ['index','rank','title','author','publisher','pubDate','description','isbn10','price','salesPoint','wperiod']

    period, book_table, information, reputation, buyc = base_df_create()

    aladin = aladin.reset_index()
    aladin = aladin.drop('index', axis = 1)
    aladin = aladin.reset_index()
    aladin['index'] = aladin['index'].astype('str')
    aladin['index'] = aladin['index'] + 'a'

    aladin['author'] = dup_count(aladin, 'author')
    aladin['author'] = aladin['author'].astype('str')
    aladin['author'] = aladin['author'] + 'a' 
    aladin['title'] = aladin['title'].str.upper()
    aladin['title'] = dup_count(aladin, 'title')
    aladin['title'] = aladin['title'].astype('str')
    aladin['title'] = aladin['title'] + aladin['index']

    aladin['wperiod'] = aladin['wperiod'].astype('str')
    aladin['wperiod'] = aladin['wperiod'] + 'w'
    aladin['wperiod'] = dup_count(aladin, 'wperiod')
    period['date'] = aladin['wperiod']
    period['pub_date'] = aladin['pubDate']
    period['year'] = aladin['wperiod'].str.slice(0,4)
    period['month'] = aladin['wperiod'].str.slice(4,6)
    period['week'] = aladin['wperiod'].str.slice(6,7)

    book_table['itemkey'] = aladin['index']
    book_table['date'] = aladin['wperiod']
    book_table['title'] = aladin['title']
    book_table['author'] = aladin['author']

    information['title'] = aladin['title']
    information['context'] = aladin['description']
    information['isbn'] = aladin['isbn10']
    information['publisher'] = aladin['publisher']

    reputation['itemkey'] = aladin['index']
    reputation['rank'] = aladin['rank']
    reputation['portal'] = 'aladin'
    reputation['portal'] = dup_count(reputation, 'portal')

    buyc['portal'] = reputation['portal']
    buyc['price'] = aladin['price']
    buyc['sales'] =  aladin['salesPoint']

    df_to_db(period, book_table, information, reputation, buyc)
    
if __name__ == "__main__":

    aladin()
    # a = datetime.datetime.now()\
    # inter_m(interpark_month('2021','05',ip_month_total, interpark_m_t))
    # inter_y(interpark_year('2021',ip_year_total, interpark_y_t))
    # kb_y(db_df_year('{}','기간',a.strftime("%Y"),kyobo_dup, kb_year)) # yes24 clear
    # kb_m(db_df_month('{}','기간',a.strftime("%Y"),a.strftime("%m"),kyobo_dup, kb_month))
    # yes_y(db_df_month('yes24_year_every','date',a.strftime("%Y"),a.strftime("%m"),yes_def, yes_year))
    # yes_d(db_df_day('yes24_day_daily','date',a.strftime("%Y"),a.strftime("%m"),a.strftime("%d"),yes_def,yes_day))
    # kb_w(db_df_week('{}','기간',a.strftime("%Y"),a.strftime("%m"),week,kyobo_dup, kb_week))
    # yes_m(db_df_week('yes24_week_weekly','date',a.strftime("%Y"),a.strftime("%m"),week,yes_def,yes_month))
    # aladin(ala_week('alading_week_every','wperiod',a.strftime("%Y"),a.strftime("%m"),week,aladina,'1'))
    # print(db_df_day('kb_monthly','기간','2022','1','',kyobo_dup, kb_month)) # kb clear
