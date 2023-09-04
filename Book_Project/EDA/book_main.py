import pandas as pd
import datetime
import pymysql
from sqlalchemy import create_engine
from book_eda import *
from book_columns import *
from datetime import timedelta

# 기존 DB를 참조할 필요가 있어 불러올때
def connet():
    con = pymysql.connect(host='localhost',
                        port=3306,
                        user='root',
                        password='lgg032800',
                        db='project3',
                        charset='utf8')
    return con

# 데이터 프레임을 DB에 넣을때
def df_to_db(period, book_table, information, reputation, buyc):

    con = pymysql.connect(host='localhost',
                            port=3306,
                            user='root',
                            password='lgg032800',
                            db='project2',
                            charset='utf8')

    engine = create_engine('mysql+pymysql://root:lgg032800@localhost/project2')

    period.to_sql('period_daily',if_exists = 'append', con = engine)
    con.commit()

    book_table.to_sql('book_table_daily',if_exists = 'append', con = engine)
    con.commit()

    information.to_sql('information_daily',if_exists = 'append', con = engine)
    con.commit()

    reputation.to_sql('reputation_daily',if_exists = 'append', con = engine)
    con.commit()

    buyc.to_sql('buyc_daily',if_exists = 'append', con = engine)
    con.commit()

def base_df_create():

    period = pd.DataFrame(index=range(0),columns = ['date','year','month','week','day','pub_date'])

    book_table = pd.DataFrame(index=range(0),columns = ['itemkey','date','title','author'])

    information = pd.DataFrame(index=range(0),columns = ['title','context','category','isbn','publisher'])

    reputation = pd.DataFrame(index=range(0),columns = ['itemkey','rank','review_num','review_rate','portal'])

    buyc = pd.DataFrame(index=range(0),columns = ['portal','accucnt','aggrcnt','price','sales'])

    return period, book_table, information, reputation, buyc


"""
데이터에 적재 되어있는 Table columns
interpark_month_grade/info/sales
column - 0 / index/ 21 / 30 /date - 0 / index / 3 /5 /15 /51 /186/187/date/rank - 0 / index /315 /319 /date
interpark_year_grade/info/sales(2020)
column - 0 / index/ 21 / 30 /date - 0 / index / 3 /5 /15 /51 /186/187/date/rank - 0 / index /315 /319 /date
jongang_isbn
column -
kb_monthly(202012)/weekly(2020205)/yearly(2020)
column - index / 0 / 기간 카테고리/ 순위 / 제목 / 저자 / 출판사 /출판연도 /평점 / 리뷰개수
yes24_day(2020-12-31)/week(2020125)/year(202012)
column - 0 / index /b_rank / context / rewiew / auther /r_date
"""

# 알라딘
def ala_week(table ,column, year, month, week, eda ,col):

    con = connet()
    cursor = con.cursor()
    # project.  ---- db 이름
    sql = f"""select * from project3.{table}
              where {column} like '%{year}{month}{week}%'"""
    cursor = con.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()

    a = pd.DataFrame(result)
    # a = a.drop(['0'], axis= 1)
    # a = a.drop(['index'], axis= 1)      
    print(a)
    return eda(a)
    
# WEEK 단위 크롤링들
def db_df_week(table ,column, year, month, week, eda ,col):

    con = connet()
    cursor = con.cursor()
    # project.  ---- db 이름
    sql = f"""select * from project3.{table}
              where {column} like '%{year}{month}{week}%'"""
    cursor = con.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()

    a = pd.DataFrame(result, columns = [col])
    # a = a.drop(['0'], axis= 1)
    a = a.drop(['index'], axis= 1)      
    print(result)
    return eda(a)

# DAY 단위 크롤링들
def db_df_day(table ,column, year, month, day, eda ,col):

    con = connet()
    cursor = con.cursor()
    # project.  ---- db 이름
    sql = f"""select * from project3.{table}
              where {column} like '%{year}-{month}-{day}%'"""
    cursor = con.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()

    a = pd.DataFrame(result, columns = [col])
    # a = a.drop(['0'], axis= 1)
    a = a.drop(['index'], axis= 1)      

    return eda(a)

# MONTH 단위 크롤링들
def db_df_month(table ,column, year, month, eda ,col):

    con = connet()
    cursor = con.cursor()
    # project.  ---- db 이름
    sql = f"""select * from project3.{table}
              where {column} like '%{year}{month}%'"""
    cursor = con.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()

    a = pd.DataFrame(result, columns = [col])
    # a = a.drop(['0'], axis= 1)
    a = a.drop(['index'], axis= 1)      

    return eda(a)

# YEAR 단위 크롤링들
def db_df_year(table ,column, year,eda ,col):

    con = connet()
    cursor = con.cursor()
    # project.  ---- db 이름
    sql = f"""select * from project3.{table}
              where {column} like '%{year}%'"""
    cursor = con.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()

    a = pd.DataFrame(result, columns = [col])
    # a = a.drop(['0'], axis= 1)
    a = a.drop(['index'], axis= 1)      

    return eda(a)

# 인터파크 일년단위
def interpark_year( year,  eda ,col):

    con = connet()
    cursor = con.cursor()
    # project.  ---- db 이름
    sql  =  f"""select * FROM project.interpark_year_grade x
                left outer join project.interpark_year_info y 
                on x.`date` = y.`date` 
                left outer join project.interpark_year_sales z 
                on x.`date` = z.`date` 
                and z.`date` = '{year}'                
                where x.21 = y.187 
                and y.187 = z.319;
                """
    cursor = con.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()

    a = pd.DataFrame(result, columns = [col])
    a = a.drop(['0'], axis= 1)
    a = a.drop(['index'], axis= 1)
    a = a.drop(['date'], axis = 1)
    a['date'] = str(year)  
    
    return eda(a)

# 인터파크 월단위
def interpark_month( year, month ,eda ,col):

    con = connet()
    cursor = con.cursor()
    # if :
    # project.  ---- db 이름
    sql  =  f"""select * FROM project.interpark_month_grade x
                left outer join project.interpark_month_info y 
                on x.`date` = y.`date` 
                left outer join project.interpark_month_sales z 
                on x.`date` = z.`date` 
                and z.`date` = '{year}{month}'
                where x.21 = y.188 
                and y.188 = z.319;
                """
    cursor = con.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()

    a = pd.DataFrame(result, columns = [col])
    a = a.drop(['0'], axis= 1)
    a = a.drop(['index'], axis= 1)
    a = a.drop(['date'], axis = 1)
    a['date'] = str(year)  
    print(eda(a))
    return eda(a)

""" columns name eda end
# yes24 eda - b_rank context review auther r_date publisher buplication - 7개
# kb eda columns - 단위 기간 카테고리 순위 제목 저자 출판사 출판연도 평점 리뷰개수 10개
# inter_y - rewview accuCnt aggrCnt author category title ProdNo rank 구매력? date 10개
# inter_m - rewview accuCnt aggrCnt author category title ProdNo rank 구매력? date 10개
# aladin - rank, title, author, publisher, pubDate, description, isbn10 price salesPoint wperiod
"""

def get_date(y, m, d):
  '''y: year(4 digits)
   m: month(2 digits)
   d: day(2 digits'''
  s = f'{y:04d}-{m:02d}-{d:02d}'
  return datetime.datetime.strptime(s, '%Y-%m-%d')

def get_week_no(y, m, d):
    target = get_date(y, m, d)
    firstday = target.replace(day=1)
    if firstday.weekday() == 6:
        origin = firstday
    elif firstday.weekday() < 3:
        origin = firstday - timedelta(days=firstday.weekday() + 1)
    else:
        origin = firstday + timedelta(days=6-firstday.weekday())
    return (target - origin).days // 7 + 1

a = datetime.datetime.now()
week = get_week_no(int(a.strftime("%Y")),int(a.strftime("%m")),int(a.strftime("%d")))

if __name__ == "__main__":
    print('ok')
    