from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
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
import time
from pprint import pprint
from datetime import datetime
from eda.news_eda_dags import *

args = {'owner':'youngdong'}


dag = DAG(dag_id = 'news_daily_crawling_batch',
          default_args=args,
          start_date= datetime(22, 11, 10),
          catchup= False,
          description= 'pipe line batch process',
          schedule_interval = '30 23 * * 1-5'
          )

t_now = datetime.now()

def joongang():
    joongang_db((t_now.strftime("%Y"),t_now.strftime("%m"),t_now.strftime("%d")))

def chosun():
    chosun_db((t_now.strftime("%Y"),t_now.strftime("%m"),t_now.strftime("%d")))

# def hani():
#     hani_db((t_now.strftime("%Y"),t_now.strftime("%m"),t_now.strftime("%d")))

def khan():
    khan_db((t_now.strftime("%Y"),t_now.strftime("%m"),t_now.strftime("%d")))

def donga():
    donga_db((t_now.strftime("%Y"),t_now.strftime("%m"),t_now.strftime("%d")))



crawling_one = PythonOperator(
	task_id = 'donga_news',
	python_callable =  joongang,
	dag = dag
)

crawling_two = PythonOperator(
	task_id = 'chosun_news',
	python_callable =  chosun,
	dag = dag
)

crawling_three = PythonOperator(
	task_id = 'khan_news',
	python_callable = khan,
	dag = dag
)

crawling_four = PythonOperator(
	task_id = 'khan_news',
	python_callable = khan_daily,
	dag = dag
)

# crawling_five = PythonOperator(
# 	task_id = 'hani_news',
# 	python_callable = hani,
# 	dag = dag
# )

crawling_one >> crawling_two >> crawling_three >> crawling_four
# >> crawling_five