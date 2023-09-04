![header](https://capsule-render.vercel.app/api?type=waving&color=auto&height=300&section=header&text=Book_Project&fontSize=90&animation=fadeIn&fontAlignY=38&desc=&descAlignY=51&descAlign=62)
<p align='center'></p>
<p align='right'>


# 프로젝트 개요
대용량의 데이터가 존재하는 필드와 분석의 가치가 있는 필드의 교집합 탐색을 위해 팀원들의 관심사를 공유하게 됨 팀원들 모두 독서를 즐기지만, 각각 선호 장르의 차이, 읽는 방식의 차이가 존재해 이에 대한 호기심을 바탕으로 국내 도서시장과 국내 소비자들을 대상으로 한 프로젝트를 기획하게 됨 

# 프로젝트의 필요성
해당 프로젝트를 통해 도서 시장에 대한 이해도를 높이며 가설검정 과정을 통해 실제로 뉴스 내 키워드 언급량과 도서 판매량의 증감이 차이가 있는지를 증명함  

# 문제정의
도서 소비자들의 여가시간 내 독서 이외의 선택지가 늘어나며 연간 독서율이 점차 우하향하는 추세 이다. 도서 시장이 이런 독서율 추세 상황에서 소비자를 확보하고, 성장을 이루기 위해서는 소비자들의 독서 문화와 그 독서 문화에 영향을 주는 요인들을 파악해야 한다.
  
# Poject Purpose of Analysis
- 도서 시장 현황 및 최근 독서 형태 현황 분석
- 소비자층 분석과 도서시장 트랜드 시각화
- 뉴스 키워드 추이와 도서 판매량 상관관계 가설 수립 및 검증
  
# Project Purpse of Development
- 배치 ELT 파이프라인 및 배포용 웹 백엔드 구현

# Team_감자탈출넘버원 팀원
- Data Engineering [#이영동](https://github.com/leeyoungdong) [#강인구](https://github.com/okok7272)  [#이진봉](https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FciOs8M%2Fbtq0oa3h0xS%2FCgClHwDFFtYq1fta4dkkw0%2Fimg.jpg)
- Data Analyst [#천세희](https://github.com/Alice1304) [#박성희](https://github.com/aurorave)
  
# 기술 스택
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Amazon S3-569A31?style=for-the-badge&logo=Amazon S3&logoColor=white"> <img src="https://img.shields.io/badge/Amazon EC2-FF9900?style=for-the-badge&logo=Amazon EC2&logoColor=white"> <img src="https://img.shields.io/badge/Apache Airflow-017CEE?style=for-the-badge&logo=Apache Airflow&logoColor=white"> <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=MySQL&logoColor=white"> <img src="https://img.shields.io/badge/Tableau-E97627?style=for-the-badge&logo=Tableau&logoColor=white"> <img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=Grafana&logoColor=white"> <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=Prometheus&logoColor=white"> 
<img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=Django&logoColor=white"> 
<img src="https://img.shields.io/badge/Amazon RDS-527FFF?style=for-the-badge&logo=Amazon RDS&logoColor=white">

# 데이터 파이프라인 (ELT)

![image](https://user-images.githubusercontent.com/100676096/206927710-9dec92f2-3053-42bc-aada-3bc8e35a220e.jpg)
- 도서/뉴스 데이터를 추출(Extract)해 Local My SQL에 적재(Load) 한후 원본데이터를 백업(innodb dump)
- 적재한 데이터를 RDS에 옮기기전 전처리와 Column을 재정의(Transform)한 후 RDS에 적재
- RDS에 적재된 데이터를 DA에서 분석 진행 후 Tableau를 사용한 BI분석 및 대시보드 생성
- RDS에 적재된 데이터는 S3스냅샷을 통해 Parquet로 2차 백업 진행
- RDS에 적재된 데이터를 통해 Django 쿼리 질의 웹 구성 및 배포
- Airflow Dag을 통한 일/ 주/ 월/ 년 단위 E/ L/ T Cronjob 진행
- Prometheus와 Grafana를 통한 데이터 작업시 컴퓨터 리소스 및 데이터 로그 시각화
  
# 데이터
# 원본 데이터 출처
![image](https://user-images.githubusercontent.com/87170837/206435454-2cee3552-334e-4e81-a36e-3528ee6e566c.png)
- 도서 베스트 셀러 데이터 - YES 24, 교보문고, 알라딘, INTERPARK (BS4 Crawling)
- 종합 뉴스 데이터 - 경향신문, 조선일보, 동아일보, 한겨례, 중앙일보 (BS4 Crawling)
- 도서 정보 데이터 - 국립중앙도서관 (API)

# Data Lake
Historical Data
![image](https://user-images.githubusercontent.com/87170837/206432241-543fc326-cb8e-4fd3-a4bf-d18804cc7bdc.png)

# Data Warehouse
Historical Data
![image](https://user-images.githubusercontent.com/87170837/206432283-2acd02c2-2594-4883-90ad-e21b832aeb10.png)

# 프로젝트 결과물
[#발표영상](https://drive.google.com/file/d/1sA9fYTPs1Uq376XMSVXJbbC0PVRwhBHb/view?usp=share_link)

[#프레젠테이션](https://github.com/leeyoungdong/Book_Project/blob/main/%5BCP2%EA%B0%90%EC%9E%90%ED%83%88%EC%B6%9C%EB%84%98%EB%B2%84%EC%9B%90_Book_Project_Final.pdf)
# BI(Business Intelligence) [클릭시 자세히 보기]


[![image](https://user-images.githubusercontent.com/87170837/206919958-838fab34-3922-46b6-b343-cdee37855828.png)](https://public.tableau.com/app/profile/.10992200/viz/shared/7N343HKXK)

# DJANGO
![image](https://user-images.githubusercontent.com/87170837/206919763-8184a5b1-f79b-43a0-81f8-20f79a56bea6.png)

![image](https://user-images.githubusercontent.com/87170837/206919786-61f703a0-ea9a-4de3-920f-e344fbef8758.png)

![image](https://user-images.githubusercontent.com/87170837/206919778-50ec62c9-c895-4a77-bed9-96143e42bc2d.png)

# AIRFLOW
![image](https://user-images.githubusercontent.com/87170837/206512889-c45e8ae0-635f-44ac-b916-a981a3614739.png)

![image](https://user-images.githubusercontent.com/87170837/206513196-47b40a8a-0bec-4a13-82d0-9f4696a75d82.png)

# GRAFANA
![image](https://user-images.githubusercontent.com/87170837/207023429-a8d9ced3-e8b6-4e06-8273-273584969ba2.png)

# 개선점
- 강인구: 한계 : django의 구현에 오래 걸려 배포를 하는 EC2로 가는 데 오랜 시간이 걸렸다. 
보완점: 해당 문제는 조금 더 먼저 파악하고 해결하지 못하여 생긴 문제로 조금 더 시간을 효율적으로 쓰지 못한 담당자인 저의 잘못입니다

- 이영동: 활용 툴을 깊게 이용하고 파이프라인에 적합하게 써보고싶다(airflow / GRAFANA DASHBOARD)
eda process가 기존 DL를 참조하여 KEY 를 생성하게 되어있어 생각보다 시간이 오래걸려 다른 KEY 설정방법을 고민해봐야할듯싶다.

- DA : 
 1. 온라인 서점 별 서로 다른 제공 데이터 범위, 아쉬운 품질 – 제공 기간, 데이터 일부 소실 등
 2. 판매’량＇데이터의 부족 – 판매지수 등 자체 제공 판매 수치 관련 데이터를 통한 다소 불완전한 분석 
 3. ‘독자＇범위의 제한 – 도서 구입 외에 구독, 대여 등 다양한 독서 경로를 반영한 추가 분석 가능성

# 프로젝트 회고
- 강인구: 이 프로젝트를 하면서 쓰지 않았던 여러 프로그램을 쓰면서 많은 것을 배운 좋은 시간이었지만, 제가 새로운 것을 배우는데 너무 많은 시간을 사용하여 다른분이 더 고생하는 문제를 발생하였습니다. 
재화님께서 말씀하신 공부하는 시간과 프로젝트를 구현하는 시간을 제대로 구분하여 한다는 말씀을 뼈저리게 이해할 수 있는 시간이었습니다.

- 박성희: 서로 다른 직군의 여럿이 한 팀을 이루어 프로젝트 전 과정을 수행하는 만큼 원활한 진행을 위해 소통과 협업이 필수적이었다는 점에서 개인적으로는 과제 진행을 위한 의사결정을 내리는 매 순간이 도전이었다. 
개별 프로젝트에서는 고민할 필요가 없었던, 공동의 목표를 달성하기 위해 지속적으로 커뮤니케이션하고 서로의 의견차를 좁혀가는 일련의 과정이 학습 측면에서나 학습 외적인 측면에서 스스로를 돌아볼 수 있는 좋은 기회였다. 
단일의 합의된 결과물을 내는 것이 프로젝트의 성공적인 완수라고 한다면, 성공적인 협업을 위해서는 ‘각자가 얼마나 많이 알고 뭘 할 수 있는지’만큼 ‘어떻게 함께 해낼 것인지’에 대한 충분한 고민과 ‘다름’에 대한 열린 자세가 필수적이라는 것도 몸소 깨닫게 되었다. 
덤으로 이번 프로젝트를 진행하며 Tableau에 조금은 친숙해질 수 있었던 점이 또 하나의 수확인데, 이번 경험을 계기로 앞으로 더 많은 데이터를 가지고 스킬을 꾸준히 키워 나가야겠다.

- 이영동: AWS의 인스턴스 비용은 비싸고, 다양한 툴을 효과적으로 사용하고 싶었는데 마음처럼 쉽지않았다. 
내가 가지고 있는 능력과 스킬을 다른사람과의 협업을 통해 성장 혹은 활용하고싶다면 먼저 내 능력에 대한 깊은이해와 의견을 정확하게 제시하고 타인의 의견을 수용할줄 아는 자세가 크게 중요하다고 생각한 프로젝트였다.

- 천세희: 코드스테이츠 7개월의 마무리를 하는 프로젝트여서 약간의 긴장을 가지고 시작했다.막히는 부분이 있을 때, 나의 컨디션과 상황에 맞춰 완급 조절할 수 있는 개인 프로젝트와 달리 
팀프로젝트는 내가 멈추면 모두가 딜레이 되는 부분이 있어서 심적으로 부담이 컸다. 팀프로젝트의 특성상 진행 과정상 매 순간 팀 내 합의가 필요했는데 그 과정을 통해 이 전의 프로젝트에서 배우지 못했던 다양한 부분들을 볼 수 있었다. 나와 같은 목적을 가지고 있는 동료를 설득하지 못한다면, 제 3자인 타인은 더 설득할 수 없기 때문에 주제선정-문제제시-분석목적을 정하는 부분에서 진통을 겪기도 했지만 결과적으로 개인프로젝트에 비해 더 설득력있는 결과물을 만들어낼수 있었다. 또한 DE파트와 프로젝트 기간 많은 시간을 함께하며 간접적으로나마 데이터 처리과정의 전반을 지켜볼 수 있었는데 이 경험이 후에 필드에서 귀하게 활용 될 것이라고 생각한다. 개인적으로 이번 프로젝트를 통해 얻은 가장 큰 수확은 DA 직무에 대한 확신이다. 가설을 세우고 검증하고, 예상했던 결과가 나왔을 때의 기쁨이 아직도 생생하다. 그 기쁨을 다시 느끼기 위해서라도 도메인지식을 늘리고, 코딩 스킬 숙련도를 쌓아야겠다고 동기부여가 되었다. 

