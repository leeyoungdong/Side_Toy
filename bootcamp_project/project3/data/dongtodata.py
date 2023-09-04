import time
# from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os

DB_FILENAME = 'project.db'
file_path = os.path.join(os.getcwd(), DB_FILENAME)
frame = ''
#마지막 데이터 줄 + 1
line = 189325
#최근 년도
days = 202003
#몇개월치 크롤링할지 정하기
month = 2

#아이뉴스
def inews(url_list, url_day, url_year, url_month):
    for i in range(line, 999999):
        try:
            #변수
            reur = requests.get(url_list[i-line])
            add_day = url_day[i-line]
            add_year = url_year[i-line]
            add_month = url_month[i-line]
            #파싱
            soup = BeautifulSoup(reur.content, 'html.parser')
            news_data = soup.select('#articleBody > p')
            data_def = re.sub('아이뉴스(.+?)기자','',str(news_data))
            data_def = preprocessing(data_def)
            # print(f'{add_year}년 {add_month}월 Day : {add_day} 데이터 : {data_def}')
            save(data_def, add_year, add_month, i-line, line)
            if i%3000 == 0:
                frame.to_csv(file_path, encoding='utf-8-sig')
                print('저장')
        except IndexError:
            frame.to_csv(file_path, encoding='utf-8-sig')
            break
#url 계산
def urls(dayss):
    # 고정변수
    url_add = 'https://www.inews24.com'
    url_list = []
    url_day = []
    url_month = []
    url_year = []

    for k in range(0,month):
        #주소 결정변수 , range가 달력(월)수
        page = 1
        day = dayss[k]
        for j in range(0,80):
            #주소결정 range가 페이지수
            pages = page+j
            url = 'https://www.inews24.com/list/it?date='+str(day)+'&page='+str(pages)+''
            #파싱
            ress = requests.get(url)
            soup = BeautifulSoup(ress.content, 'html.parser')
            for i in range(0,30):
                #range가 주소 저장수
                try:
                    data = soup.select('body > main > article > ol > li:nth-child('+str(i+1)+') > div > a')[0].get('href')
                    url_list.append(url_add+data)
                    url_day.append(day)
                    url_year.append(day//100)
                    url_month.append(months(day))
                    print(f'저장년도 : {url_day[-1]}, 저장주소 : {url_list[-1]}')
                except IndexError:
                    break
    #중복제거
    url_list = set(url_list)
    url_list = list(url_list)
    inews(url_list, url_day, url_year, url_month)
#전처리
def preprocessing(data):
    data_def = re.sub('<(.+?)>', '',str(data))
    data_def = re.sub('\r', '', str(data_def))
    data_def = re.sub('\t', '', str(data_def))
    data_def = re.sub('\n', '', str(data_def))
    data_def = re.sub('\f', '', str(data_def))
    data_def = re.sub('\v', '', str(data_def))
    data_def = re.sub('\[', '', str(data_def))
    data_def = re.sub('\]', '', str(data_def))
    data_def = data_def.strip()
    return data_def
#날짜경로계산
def day():
    global days
    days_list = []
    for i in range(0,month):
        if days%100 == 1:
            days = days-89
        else:
            days = days-1
        days_list.append(days)
    return days_list
#월 계산
def months(month):
    data = month-202200
    while data < 0:
        data = data+100
    return data
#저장
def save(in_data, year, month, add_line, lines):
    global data_count, frame
    #데이터 추가
    frame.loc[add_line+line, 'text'] = in_data
    #년도 추가
    frame.loc[add_line+line, 'year'] = str(year)
    frame.loc[add_line+line, 'month'] = str(month)
    # frame.to_csv(file_path, encoding='utf-8-sig')
#데이터프레임(csv) 불러오기
def dataframe():
    global frame
    try:
        frame = pd.read_csv(file_path, index_col = 0)
        frame = frame.astype({'year': 'str', 'month': 'str'})
    except:
        data = {
            'text' : [],
            'year' : [],
            'month' : []
        }
        frame = pd.DataFrame(data)
        frame.to_csv(file_path, encoding='utf-8-sig')
    return frame
    
if __name__ == '__main__' :
    dataframe()
    print(frame)
    urls(day())
    print(frame)