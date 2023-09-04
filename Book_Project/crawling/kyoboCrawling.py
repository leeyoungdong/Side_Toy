from sqlalchemy import create_engine
import pandas as pd
import re
import urllib.request
import requests
import json

period_nums = ['002', '003', '004']
#002 : 주간 003 :월간 004 :연간

def kyobo_week(year, month, week ):

    unit_size = '주간'
    syear = str(year)
    smonth = str(month).zfill(2)
    sweek =str(week)

    category = '종합'
    table_name = 'kyobo_all_week'
    df = pd.DataFrame(columns=[ '기간', '카테고리', '순위', '제목', '저자', '출판사', '출판연도', '평점', '리뷰개수'])

    engine = create_engine("mysql+pymysql://root:0000@localhost/kyobobook", encoding= 'utf-8')
    conn = engine.connect()
    unit_period = syear+smonth+sweek

    response = requests.get(f'https://product.kyobobook.co.kr/api/gw/pub/pdt/best-seller/total?page=1&per=200&period=002&ymw={unit_period}&bsslBksClstCode=A').json()
    for k in range(0, 200):
        rank = k
    
        try:
            rank = response['data']['bestSeller'][k]['prstRnkn']
    
        except:
            break
    
        title = response['data']['bestSeller'][k]['cmdtName']
        author = response['data']['bestSeller'][k]['chrcName']
        publishor = response['data']['bestSeller'][k]['pbcmName']
        published_year = response['data']['bestSeller'][k]['rlseDate']
        review_score = response['data']['bestSeller'][k]['buyRevwRvgr']
        review_num = response['data']['bestSeller'][k]['buyRevwNumc']
        temp_df = pd.DataFrame({ '기간' : unit_period, '카테고리' : category, '순위' : rank, '제목' : title, '저자' : author, '출판사' : publishor,'출판연도' : published_year, '평점' : review_score, '리뷰개수' : review_num}, index=[0])
        df= pd.concat([df,temp_df])

    df = df.drop_duplicates()
    df.to_sql(name=table_name , con= engine, if_exists= 'append', index=False)
    conn.close()

def kyobo_month(year,month):

    unit_size = '월간'
    syear = str(year)
    smonth = str(month).zfill(2)

    category = '종합'
    table_name = 'kyobo_all_month'
    df = pd.DataFrame(columns=[ '기간', '카테고리', '순위', '제목', '저자', '출판사', '출판연도', '평점', '리뷰개수'])

    engine = create_engine("mysql+pymysql://root:0000@localhost/kyobobook", encoding= 'utf-8')
    conn = engine.connect()
    unit_period = syear + smonth

    response = requests.get(f'https://product.kyobobook.co.kr/api/gw/pub/pdt/best-seller/total?page=1&per=200&period=003&ymw={unit_period}&bsslBksClstCode=A').json()
    
    for k in range(0, 200):
        rank = k
    
        try:
            rank = response['data']['bestSeller'][k]['prstRnkn']
    
        except:
            break
    
        title = response['data']['bestSeller'][k]['cmdtName']
        author = response['data']['bestSeller'][k]['chrcName']
        publishor = response['data']['bestSeller'][k]['pbcmName']
        published_year = response['data']['bestSeller'][k]['rlseDate']
        review_score = response['data']['bestSeller'][k]['buyRevwRvgr']
        review_num = response['data']['bestSeller'][k]['buyRevwNumc']

        temp_df = pd.DataFrame({'기간' : unit_period, '카테고리' : category, '순위' : rank, '제목' : title, '저자' : author, '출판사' : publishor,'출판연도' : published_year, '평점' : review_score, '리뷰개수' : review_num}, index=[0])
        df= pd.concat([df,temp_df])

    df = df.drop_duplicates()
    df.to_sql(name=table_name , con= engine, if_exists= 'append', index=False)
    conn.close()

def kyobo_year(year):


    category = '종합'
    table_name = 'kyobo_all_year'
    df = pd.DataFrame(columns=[ '기간', '카테고리', '순위', '제목', '저자', '출판사', '출판연도', '평점', '리뷰개수'])

    engine = create_engine("mysql+pymysql://root:0000@localhost/kyobobook", encoding= 'utf-8')
    conn = engine.connect()
   
    syear = str(year)
    unit_period =syear
    response = requests.get(f'https://product.kyobobook.co.kr/api/gw/pub/pdt/best-seller/total?page=1&per=200&period=004&ymw={unit_period}&bsslBksClstCode=A').json()
    
    for k in range(0, 200):
        rank = k
    
        try:
            rank = response['data']['bestSeller'][k]['prstRnkn']
    
        except:
            break
    
        title = response['data']['bestSeller'][k]['cmdtName']
        author = response['data']['bestSeller'][k]['chrcName']
        publishor = response['data']['bestSeller'][k]['pbcmName']
        published_year = response['data']['bestSeller'][k]['rlseDate']
        review_score = response['data']['bestSeller'][k]['buyRevwRvgr']
        review_num = response['data']['bestSeller'][k]['buyRevwNumc']
        temp_df = pd.DataFrame({ '기간' : unit_period, '카테고리' : category, '순위' : rank, '제목' : title, '저자' : author, '출판사' : publishor,'출판연도' : published_year, '평점' : review_score, '리뷰개수' : review_num}, index=[0])
        df= pd.concat([df,temp_df])

        df = df.drop_duplicates()
        df.to_sql(name=table_name , con= engine, if_exists= 'append', index=False)
    conn.close()


# def unit_dicision(period_num):
#     week_list = [2022112, 2022111, 2022105, 2022104, 2022103, 2022102, 2022101, 2022094, 2022093, 2022092, 2022091, 2022084, 2022083, 2022082, 2022081, 2022075, 2022074, 2022073, 2022072, 2022071, 2022064, 2022063, 2022062, 2022061, 2022054, 2022053, 2022052, 2022051, 2022045, 2022044, 2022043, 2022042, 2022041,2022034,2022033,2022032,2022031, 2022024, 2022023, 2022022, 2022021, 2022015, 2022014, 2022013, 2022012, 2022011, 2021124, 2021123, 2021122, 2021121, 2021114, 2021113,2021112]
#     month_list =[202210, 202209, 202208, 202207, 202206, 202205, 202204, 202203, 202202, 202201, 202112]
#     year_list = [2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008]
#     unit_size = '주간'
#     temp_list = []
#     c_table_name = 'all'
#     l_table_name = 'list'
#     u_table_name = '_week_'
#     if period_num =='002':
#         unit_size = '주간'
#         temp_list =week_list
#         u_table_name = '_week_'

#     if period_num == '003':
#         unit_size = '월간'
#         temp_list = month_list
#         u_table_name = '_month_'

#     if period_num == '004':

#         unit_size = '연간'
#         temp_list = year_list
#         u_table_name = '_year_'

#     category = '종합'
#     c_table_name = 'all'
#     table_name = c_table_name + u_table_name + l_table_name
#     df = pd.DataFrame(columns=['단위', '기간', '카테고리', '순위', '제목', '저자', '출판사', '출판연도', '평점', '리뷰개수'])
#     engine = create_engine("mysql+pymysql://root:0000@localhost/kyobobook", encoding= 'utf-8')
#     conn = engine.connect()
#     for unit_period in temp_list:
#         print(unit_period)
#         response = requests.get(f'https://product.kyobobook.co.kr/api/gw/pub/pdt/best-seller/total?page=1&per=200&period=003&ymw={unit_period}&bsslBksClstCode=A').json()
#         for k in range(0, 200):
#             rank = k
#             try:
#                 rank = response['data']['bestSeller'][k]['prstRnkn']
#             except:
#                 break
#             title = response['data']['bestSeller'][k]['cmdtName']
#             author = response['data']['bestSeller'][k]['chrcName']
#             publishor = response['data']['bestSeller'][k]['pbcmName']
#             published_year = response['data']['bestSeller'][k]['rlseDate']
#             review_score = response['data']['bestSeller'][k]['buyRevwRvgr']
#             review_num = response['data']['bestSeller'][k]['buyRevwNumc']
#             temp_df = pd.DataFrame({'단위' : unit_size, '기간' : unit_period, '카테고리' : category, '순위' : rank, '제목' : title, '저자' : author, '출판사' : publishor,'출판연도' : published_year, '평점' : review_score, '리뷰개수' : review_num}, index=[0])
#             df= pd.concat([df,temp_df])

#         df = df.drop_duplicates()
#         df.to_sql(name=table_name , con= engine, if_exists= 'append', index=False)
#     conn.close()

if __name__ == '__main__':
    #period num은 위의 데이터 참고 002: 주간 003: 월간 004 : 연간
   print('1')





