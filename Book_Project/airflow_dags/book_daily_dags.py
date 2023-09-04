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
from book.yes24 import yes24_day

args = {'owner':'youngdong'}


dag = DAG(dag_id = 'book_daily_batch',
          default_args=args,
          start_date= datetime(22, 11, 10),
          catchup= False,
          description= 'pipe line batch process',
          schedule_interval = '30 23 * * *'
          )

t_now = datetime.now()

def yes24():
     for k in range(1,11):  yes24_day(t_now.strftime("%Y"),t_now.strftime("%m"),t_now.strftime("d"),k)


crawling_one = PythonOperator(
	task_id = 'yes24_daily',
	python_callable =  yes24,
	dag = dag
)


crawling_one
