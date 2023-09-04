import pandas as pd
import numpy as np
from datetime import timedelta
# interpark grade
def ip_grade(df):

    df = df.rename(columns={'21': 'ProdNo', '30': 'review'})

    df['review'] = df['review'].str.replace(',','').astype(np.float64)
    df = df.drop_duplicates(keep = 'first')
    
    return df

# interpark info
def ip_m_info(df):

    df = df.rename(columns={'3': 'accuCnt', '5': 'aggrCnt', '15': 'author', '51': 'category', '187': 'title', '188': 'ProdNo'})

    df.author = df.author.str.replace('\\','')
    df.author = df.author.str.replace('/','')
    df.category = df.category.str.replace('\\','')
    df.category = df.category.str.replace('/','')
    df['accuCnt'] = df['accuCnt'].str.replace(',','').astype(np.float64)
    df['aggrCnt'] = df['aggrCnt'].str.replace(',','').astype(np.float64)
    df.author = df.author.str.replace("'", "")
    df.author = df.author.str.replace('"', '')
    df.category = df.category.str.replace("'", "")
    df.title = df.title.str.replace("'", "")
    df.title = df.title.str.replace('"', '')
    df.title = df.title.str.replace('\\','')
    df.title = df.title.str.replace(' ','')
    df.ProdNo = df.ProdNo.str.replace('\\','')
    df.ProdNo = df.ProdNo.str.replace(' ','')
    df = df.drop_duplicates(keep = 'first')
    
    return df

# interpark info
def ip_y_info(df):

    df = df.rename(columns={'3': 'accuCnt', '5': 'aggrCnt', '15': 'author', '51': 'category', '186': 'title', '187': 'ProdNo'})

    df.author = df.author.str.replace('\\','')
    df.author = df.author.str.replace('/','')
    df.category = df.category.str.replace('\\','')
    df.category = df.category.str.replace('/','')
    df['accuCnt'] = df['accuCnt'].str.replace(',','').astype(np.int64)
    df['aggrCnt'] = df['aggrCnt'].str.replace(',','').astype(np.int64)
    df.author = df.author.str.replace("'", "")
    df.author = df.author.str.replace('"', '')
    df.category = df.category.str.replace("'", "")
    df.title = df.title.str.replace("'", "")
    df.title = df.title.str.replace('"', '')
    df.title = df.title.str.replace('\\','')
    df.title = df.title.str.replace(' ','')
    df.ProdNo = df.ProdNo.str.replace('\\','')
    df.ProdNo = df.ProdNo.str.replace(' ','')
    df = df.drop_duplicates(keep = 'first') # 27975 - 1530 = 26445
    
    return df

# interpark sales
def ip_sales(df):

    df = df.rename(columns={'315': '구매력?', '319': 'ProdNo'})

    df['구매력?'] = df['구매력?'].str.replace(',','').astype(np.int64)
    df = df.drop_duplicates(keep = 'first')
    
    return df

# interpark year total eda
def ip_year_total(df):

    df.columns = ['21','30','3','5','15','51','186','187','rank','315','319','date'] 
    df = df.rename(columns={'21': '0', '30': 'review','3': 'accuCnt', '5': 'aggrCnt', '15': 'author', '51': 'category', '186': 'title', '187': 'ProdNo','315': '구매력?', '319': '0'})

        # df['ProdNo'] = df['ProdNo'].apply(lambda x: x['ProdNo'].str.replace('\\',''), axis = 1)
    df['구매력?'] = df['구매력?'].str.replace(',','').astype(np.int64)
    df = df.drop_duplicates(keep = 'first')
    df.author = df.author.str.replace('\\','')
    df.author = df.author.str.replace('/','')
    df.category = df.category.str.replace('\\','')
    df.category = df.category.str.replace('/','')
    df['accuCnt'] = df['accuCnt'].str.replace(',','').astype(np.int64)
    df['aggrCnt'] = df['aggrCnt'].str.replace(',','').astype(np.int64)
    df.author = df.author.str.replace("'", "")
    df.author = df.author.str.replace('"', '')
    df.category = df.category.str.replace("'", "")
    df.title = df.title.str.replace("'", "")
    df.title = df.title.str.replace('"', '')
    df.title = df.title.str.replace('\\','')
    df.title = df.title.str.replace(' ','')
    df.ProdNo = df.ProdNo.str.replace('\\','')
    # df['ProdNo'] = df['ProdNo'].apply(lambda x: x['ProdNo'].str.replace('\\',''), axis = 1)
    df.ProdNo = df.ProdNo.str.replace(' ','')
    # df['ProdNo'] = df['ProdNo'].apply(lambda x: x['ProdNo'].str.replace(' ',''), axis = 1)    
    df = df.drop_duplicates(keep = 'first') # 27975 - 1530 = 26445
    # df = df.rename(columns={'21': 'ProdNo', '30': 'review'})
    df['review'] = df['review'].str.replace(',','').astype(np.float64)
    df = df.drop_duplicates(keep = 'first')
    df = df.drop(['0'], axis= 1)    
    
    return df
# interpark month total eda
def ip_month_total(df):

    # df.columns = ['review','accuCnt','aggrCnt','author','category','title','ProdNo','rank', '구매력?',  'date'] 
    df.columns = ['21','30','3','5','15','51','187','188','rank','315','319','date'] 
    # df.columns = [ '0', 'review','accuCnt', 'aggrCnt',  'author','category','title', 'rank', 'ProdNo', '구매력?',  '0','date']
    df = df.rename(columns={'21': '0', '30': 'review','3': 'accuCnt', '5': 'aggrCnt', '15': 'author', '51': 'category', '187': 'title', '188': 'ProdNo','315': '구매력?', '319': '0'})

    # df['ProdNo'] = df['ProdNo'].apply(lambda x: x['ProdNo'].str.replace('\\',''), axis = 1)
    # df['구매력?'] = df['구매력?'].str.replace(',','').astype(np.int64)
    df = df.drop(['0'], axis= 1) 
    # print(df)
    df = df.drop_duplicates(keep = 'first')
    df.author = df.author.str.replace('\\','')
    df.author = df.author.str.replace('/','')
    df.category = df.category.str.replace('\\','')
    df.category = df.category.str.replace('/','')
    df['accuCnt'] = df['accuCnt'].str.replace(',','').astype(np.int64)
    df['aggrCnt'] = df['aggrCnt'].str.replace(',','').astype(np.int64)
    df.author = df.author.str.replace("'", "")
    df.author = df.author.str.replace('"', '')
    df.category = df.category.str.replace("'", "")
    df.title = df.title.str.replace("'", "")
    df.title = df.title.str.replace('"', '')
    df.title = df.title.str.replace('\\','')
    df.title = df.title.str.replace(' ','')
    df.ProdNo = df.ProdNo.str.replace('\\','')
    # df['ProdNo'] = df['ProdNo'].apply(lambda x: x['ProdNo'].str.replace('\\',''), axis = 1)
    df.ProdNo = df.ProdNo.str.replace(' ','')
    # df['ProdNo'] = df['ProdNo'].apply(lambda x: x['ProdNo'].str.replace(' ',''), axis = 1)    
    df = df.drop_duplicates(keep = 'first') # 27975 - 1530 = 26445
    df = df.rename(columns={'21': 'ProdNo', '30': 'review'})
    df['review'] = df['review'].str.replace(',','').astype(np.float64)
    df = df.drop_duplicates(keep = 'first')
    # df = df.drop(['0'], axis= 1)    

    return df
# 교보 EDA
def kyobo_dup(df):

    df.columns =  ['단위','기간', '카테고리','순위', '제목','저자', '출판사', '출판연도', '평점', '리뷰개수']

    df = df.drop_duplicates(subset = ['기간', '순위', '제목']) # keep = 'first' by default

    return df 
# YES24 EDA
def yes_def(yes_year):

    yes_year.columns = ['b_rank', 'context', 'review','auther', 'r_date']

    yes_year = yes_year.replace(r'\n','', regex=True)
    yes_year['context'] = yes_year['context'].replace(r'\r','', regex=True)
    yes_year['review'] = yes_year['review'].replace(r'\r','', regex=True)
    yes_year['auther'] = yes_year['auther'].replace(r'\r','', regex=True)
    yes_year['r_date'] = yes_year['r_date'].replace(r'\r','', regex=True)
    yes_year[['auther','publisher','publication']] = yes_year['auther'].str.split('|', n=2, expand = True)
    yes_year['publisher'] = yes_year['publisher'].replace(' ','', regex=True)
    yes_year['context'] = yes_year['context'].str.slice(4)

    return yes_year
# 알라딘 EDA
def aladina(df):

    df.drop_duplicates(inplace=True)
    df = df.dropna()

    return df

# groupby로 데이터 프레임 각행에 중복되는 숫자 붙혀주는 함수식
def dup_count(table, column):
    a = table.groupby(f'{column}').cumcount()+1
    table['new1'] = a
    table['new1'] = table['new1'].astype('str')
    table[f'{column}'] = table[f'{column}'] + '/' +table['new1']
    return table[f'{column}']