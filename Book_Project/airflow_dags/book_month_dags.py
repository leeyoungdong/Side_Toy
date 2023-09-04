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
from books.yes24 import yes24_year
from books.interpark import interpark_month

args = {'owner':'youngdong'}


dag = DAG(dag_id = 'book_month_batch',
          default_args=args,
          start_date= datetime(22, 11, 10),
          catchup= False,
          description= 'pipe line batch process',
          schedule_interval = '30 23 2 * *'
          )

t_now = datetime.now()

def yes24_month():
     for i in range(1,11): yes24_year(t_now.strftime("%Y"),t_now.strftime("%m"),i)

def interpark_m():
	for k in range(1,6): interpark_month(t_now.strftime("%Y"),t_now.strftime("%m"), k)

yes24_dag = PythonOperator(
	task_id = 'yes24_month',
	python_callable =  yes24_month,
	dag = dag
)

interpark_mon = PythonOperator(
	task_id = 'interaprk_m',
	python_callable = interpark_m,
	dag = dag
)

yes24_dag >> interpark_mon
