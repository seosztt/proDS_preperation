# -*- coding: utf-8 -*-
"""
Created on 2021

@author: Administrator
"""

#%%

# =============================================================================
# =============================================================================
# # 문제 06 유형(DataSet_06.csv 이용)
#
# 구분자 : comma(“,”), 4,323 Rows, 19 Columns, UTF-8 인코딩

# 주택 관련 정보를 바탕으로 주택 가격을 예측해 보고자 한다. 
# 다음은 확보한 주택 관련 데이터로 총 19개 컬럼으로 구성되어
# 있다.

# 컬 럼 / 정 의 / Type
# id / 매물 번호 / Double
# date / 날짜 / String
# price / 거래 가격 / Double
# bedrooms / 방 개수 / Double
# bathrooms / 화장실 개수 (화장실은 있으나 샤워기 없는 경우 0.5로 처리) / Double
# sqft_living / 건축물 면적 / Double
# sqft_lot / 대지 면적 / Double
# floors / 건축물의 층수 / Double
# waterfront / 강변 조망 가능 여부 (0 / 1) / Double
# view / 경관 (나쁨에서 좋음으로 0 ~ 4로 표시) / Double
# condition / 관리 상태 (나쁨에서 좋음으로 1 ~ 5로 표시) / Double
# grade / 등급 (낮음에서 높음으로 1 ~ 13으로 표시) / Double
# sqft_above / 지상 면적 / Double
# sqft_basement / 지하실 면적 / Double
# yr_built / 건축 연도 / Double
# yr_renovated / 개축 연도 / Double
# zipcode / 우편번호 / Double
# sqft_living15 / 15개의 인근 주택의 평균 건물 면적 / Double
# sqft_lot15 / 15개의 인근 주택의 평균 대지 면적 / Double
# =============================================================================
# =============================================================================

import pandas as pd
data6=pd.read_csv('Dataset_06.csv')

#%%

# =============================================================================
# 1.강변 조망이 가능한지 여부(waterfront)에 따라 평균 주택 가격을 계산하고 조망이
# 가능한 경우와 그렇지 않은 경우의 평균 가격 차이의 절대값을 구하시오. 답은
# 소수점 이하는 버리고 정수부만 기술하시오. (답안 예시) 1234567
# =============================================================================

data6[data6['waterfront']==1]['price'].mean() - data6[data6['waterfront']==0]['price'].mean()




# 답: 1167272

#%%

# =============================================================================
# 2.price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, yr_built 등 7개의 변수 간의
# 상관분석을 수행하고 price와의 상관계수의 절대값이 가장 큰 변수와 가장 작은
# 변수를 차례로 기술하시오. (답안 예시) view, zipcode
# 
# =============================================================================

data6.columns

data6[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'yr_built']].corr()


# 답: sqft_living, yr_built


#%%

# =============================================================================
# 3. id, date, 그리고 zipcode를 제외한 모든 변수를 독립변수로, price를 종속변수로 하여
# 회귀분석을 수행하시오. 통계적 유의성을 갖지 못하는 독립변수를 제거하면 회귀
# 모형에 남는 변수는 모두
# 몇 개인가? 이 때 음의 회귀계수를 가지는 변수는 몇 개인가? (답안 예시) 5, 3
# =============================================================================

# =============================================================================
# (참고)
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from statsmodels.formula.api import ols
# =============================================================================

q3=data6.copy()
# q3=q3.drop(['id', 'date', 'zipcode'], axis=1)

q3.columns
x_list = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
       'sqft_living15', 'sqft_lot15']
from statsmodels.formula.api import ols

formula = 'price~' + '+'.join(x_list)
ols1 = ols(formula, q3).fit()
ols1.summary()

q3_out=ols1.pvalues.drop('Intercept')
q3_out2= q1_out[q1_out < 0.05].index
len(q3_out2)

len(q3_out2[ols1.params[q3_out2] < 0])
sum(ols1.params[q3_out2] < 0)
# 답: 13, 2
#%%

# =============================================================================
# =============================================================================
# # 문제 07 유형(DataSet_07.csv 이용)
#
# 구분자 : comma(“,”), 400 Rows, 9 Columns, UTF-8 인코딩
#
# 대학원 진학을 위하여 어떤 항목이 중요하게 영향을 미치는지
# 아래 데이터로 분석하고자 한다.

# 컬 럼 / 정 의 / Type
# Serial_No / 구분자 번호 / Double
# GRE / GRE 시험 성적 / Double
# TOEFL / TOEFL 시험 성적 / Double
# University_Rating / 대학 평가 그룹 (1 ~ 5) / Double
# SOP / 자기 소개서 점수 (1 ~ 5) / Double
# LOR / 추천서 점수 (1 ~ 5) / Double
# CGPA / 학부 평량 평점 (10점 만점 환산 점수) / Double
# Research / 연구 참여 경험 여부 (0 / 1) / Double
# Chance_of_Admit / 합격 가능성 / Double
# =============================================================================
# =============================================================================

# =============================================================================
# (참고)
# #1
# import pandas as pd
# #2
# import scipy.stats as stats
# #3
# from sklearn.linear_model import LogisticRegression
# Solver = ‘liblinear’, random_state = 12
# =============================================================================
import pandas as pd
data7 = pd.read_csv('Dataset_07.csv')
data7.info()
#%%

# =============================================================================
# 1. 합격 가능성에 GRE, TOEFL, CGPA 점수 가운데 가장 영향이 큰 것이 어떤 점수인지
# 알아 보기 위해서 상관 분석을 수행한다.
# - 피어슨(Pearson) 상관계수 값을 구한다.
# - Chance_of_Admit와의 가장 큰 상관계수 값을 가지는 항목의 상관계수를 소수점 넷째
# 자리에서 반올림하여 셋째 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================

q1=data7.copy()
q1[['GRE', 'TOEFL', 'CGPA', 'Chance_of_Admit']].corr()['Chance_of_Admit']

# 답: 0.8733

#%%

# =============================================================================
# 2.GRE 점수의 평균 이상을 받은 그룹과 평균 미만을 받은 그룹의 CGPA 평균은 차이가
# 있는지
# 검정을 하고자 한다.
# - 적절한 검정 방법을 선택하고 양측 검정을 수행하시오 (등분산으로 가정)
# - 검정 결과, 검정통계량의 추정치를 소수점 셋째 자리에서 반올림하여 소수점 두 자리까지
# 기술하시오.
# (답안 예시) 1.23
# =============================================================================

q2=data7.copy()
GRE_m=q2.GRE.mean()

group0=q2[q2.GRE < GRE_m]['CGPA']
group1=q2[q2.GRE >= GRE_m]['CGPA']

from scipy.stats import ttest_ind

t, p = ttest_ind(group1, group0, equal_var=True)
t
# 답: 19.44 


#%%

# =============================================================================
# 3.Chance_of_Admit 확률이 0.5를 초과하면 합격으로, 이하이면 불합격으로 구분하고
# 로지스틱 회귀분석을 수행하시오.
# - 원데이터만 사용하고, 원데이터 가운데 Serial_No와 Label은 모형에서 제외
# - 각 설정값은 다음과 같이 지정하고, 언급되지 않은 사항은 기본 설정값을 사용하시오
# Seed : 123
# - 로지스틱 회귀분석 수행 결과에서 로지스틱 회귀계수의 절대값이 가장 큰 변수와 그 값을
# 기술하시오. 
# (로지스틱 회귀계수는 반올림하여 소수점 둘째 자리까지 / Intercept는 제외)
# (답안 예시) abc, 0.12
# =============================================================================

import numpy as np

q3=data7.copy()
q3['Label'] = np.where(q3.Chance_of_Admit > 0.5, 1, 0)

X=q3.drop(['Serial_No','Label', 'Chance_of_Admit'], axis=1)
y=q3['Label']

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(fit_intercept=False, random_state=123)
lr.fit(X, y)

q3_out = pd.Series(dict(zip(X.columns,lr.coef_[0]))).abs()
q3_out.idxmax()
q3_out.nlargest(1)

# 답: CGPA, 2.00
#%%

# =============================================================================
# =============================================================================
# # 문제 08 유형(DataSet_08.csv 이용)
#
# 구분자 : comma(“,”), 50 Rows, 5 Columns, UTF-8 인코딩
#
# 스타트업 기업들의 수익성에 대한 분석을 하기 위하여
# 아래와 같은 데이터를 입수하였다
#
# 
# 컬 럼 / 정 의 / Type
# RandD_Spend / 연구개발비 지출 / Double
# Administration / 운영관리비 지출 / Double
# Marketing_Spend / 마케팅비 지출 / Double
# State / 본사 위치 / String
# Profit / 이익 / Double
# =============================================================================
# =============================================================================

# =============================================================================
# (참고)
# #1
# import pandas as pd
# import numpy as np
# #3
# from sklearn.linear_model import LinearRegression
# =============================================================================

#%%
import pandas as pd
data8=pd.read_csv('Dataset_08.csv')
data8.info()
# =============================================================================
# 1.각 주(State)별 데이터 구성비를 소수점 둘째 자리까지 구하고, 알파벳 순으로
# 기술하시오(주 이름 기준).
# (답안 예시) 0.12, 0.34, 0.54
# =============================================================================
data8.State.unique()
data8.groupby('State').count()

# 답: 0.34, 0.32, 0.34

#%%

# =============================================================================
# 2.주별 이익의 평균을 구하고, 평균 이익이 가장 큰 주와 작은 주의 차이를 구하시오. 
# 차이값은 소수점 이하는 버리고 정수부분만 기술하시오. (답안 예시) 1234
# =============================================================================

q2=data8.copy()


q2[q2.State == 'California']['Profit'].mean()
q2[q2.State == 'New York']['Profit'].mean()
q2[q2.State == 'Florida']['Profit'].mean()
q2[q2.State == 'Florida']['Profit'].mean() - q2[q2.State == 'California']['Profit'].mean()

# 답: 14868
#%%

# =============================================================================
# 3.독립변수로 RandD_Spend, Administration, Marketing_Spend를 사용하여 Profit을 주별로
# 예측하는 회귀 모형을 만들고, 이 회귀모형을 사용하여 학습오차를 산출하시오.
# - 주별로 계산된 학습오차 중 MAPE 기준으로 가장 낮은 오차를 보이는 주는 어느
# 주이고 그 값은 무엇인가? (반올림하여 소수점 둘째 자리까지 기술하시오)
# - (MAPE = Σ ( | y - y ̂ | / y ) * 100/n )
# (답안 예시) ABC, 1.56
# =============================================================================

q3=data8.copy()
from sklearn.linear_model import LinearRegression

X_list=['RandD_Spend', 'Administration', 'Marketing_Spend']

State_list = q3.State.unique()
result=[]
for i in State_list:
    temp=q3[q3.State == i]
    lm=LinearRegression().fit(temp[X_list], temp.Profit)
    pred=lm.predict(temp[X_list])
    out=((abs(temp.Profit - pred) / temp.Profit)*100/len(pred)).sum()
    result.append([i,out])
result

# 답: Florida, 5.71
#%%

# =============================================================================
# =============================================================================
# # 문제 09 유형(DataSet_09.csv 이용)
#
# 구분자 : comma(“,”), 2000 Rows, 16 Columns, UTF-8 인코딩
#
# 항공사에서 고객만족도 조사를 하고 서비스 개선에 활용하고자
# 아래와 같은 데이터를 준비하였다.
#
# 컬 럼 / 정 의 / Type
# satisfaction / 서비스 만족 여부 / String
# Gender / 성별 / String
# Age / 나이 / Double
# Customer_Type / 고객 타입 / String
# Class / 탑승 좌석 등급 / String
# Flight_Distance / 비행 거리 / Double
# Seat_comfort / 좌석 안락도 점수 / Double
# Food_and_Drink / 식사와 음료 점수 / Double
# Inflight_wifi_service / 기내 와이파이 서비스 점수 / Double
# Inflight_entertainment / 기내 엔터테인먼트 서비스 점수 / Double
# Onboard_service / 탑승 서비스 점수 / Double
# Leg_room_service / 다리 공간 점수 / Double
# Baggage_handling / 수하물 취급 점수 / Double
# Cleanliness / 청결도 점수 / Double
# Departure_Daly_in_Minutes / 출발 지연 (분) / Double
# Arrival_Delay_in_Minutes / 도착 지연 (분) / Double
# =============================================================================
# =============================================================================

import pandas as pd
data9=pd.read_csv('Dataset_09.csv')
data9.info()
# =============================================================================
# (참고)
# #1
# import pandas as pd
# import numpy as np
# #2
# import scipy.stats as stats
# #3
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# 
# =============================================================================

#%%

# =============================================================================
# 1.데이터 타입을 위 표에 정의된 타입으로 전처리를 한 후, 데이터 파일 내에 결측값은
# 총 몇 개인가? (답안 예시) 1
# =============================================================================

data9.isna().sum().sum()

# 답: 5




#%%
# =============================================================================
# 2.다음에 제시된 데이터 처리를 하고 카이제곱 독립성 검정을 수행하시오.
# - 결측값이 있다면 해당 행을 제거하시오.
# - 나이는 20 이하이면 10, 30 이하이면 20, 40 이하이면 30, 50 이하이면 40, 60 이하이면 50, 
# 60 초과는 60으로 변환하여 Age_gr으로 파생변수를 생성하시오.
# - Age_gr, Gender, Customer_Type, Class 변수가 satisfaction에 영향이 있는지 카이제곱
# 독립성 검정을 수행하시오. 
# - 연관성이 있는 것으로 파악된 변수의 검정통계량 추정치를 정수 부분만 기술하시오. 
# (답안 예시) 123
# =============================================================================

q2=data9.copy()

q2=q2.dropna()
q2.columns

import numpy as np
q2['Age_gr']=np.where(q2.Age <=20, 10,
                      np.where(q2.Age <=30, 20,
                               np.where(q2.Age <=40, 30,
                                        np.where(q2.Age <=50, 40,
                                                 np.where(q2.Age <=60, 50, 60)))))
q2.satisfaction

var_list=['Age_gr', 'Gender', 'Customer_Type', 'Class']
from scipy.stats import chi2_contingency
result = []
for i in var_list:
    temp=pd.crosstab(q2[i], q2['satisfaction'])
    chi2, p, dof, ef = chi2_contingency(temp, correction = False)
    result.append([i, chi2, p])
result

# 답: 1069

#%%

# =============================================================================
# 3.고객 만족도를 라벨로 하여 다음과 같이 로지스틱 회귀분석을 수행하시오. 
# - 결측치가 포함된 행은 제거
# - 데이터를 7대 3으로 분리 (Seed = 123)
# - 아래의 11개 변수를 Feature로 사용
# Flight_Distance, Seat_comfort, Food_and_drink, Inflight_wifi_service, 
# Inflight_entertainment,Onboard_service, Leg_room_service, Baggage_handling,
# Cleanliness, Departure_Delay_in_Minutes, Arrival_Delay_in_Minutes
# 
# - Seed = 123, 이외의 항목은 모두 Default 사용
# - 예측 정확도를 측정하고 dissatisfied의 f1 score를 소수점 넷째 자리에서 반올림하여
# 소수점 셋째 자리까지 기술하시오. (답안 예시) 0.123
# =============================================================================
from sklearn.model_selection import train_test_split
q3=data9.copy()
q3=q3.dropna()

X_list = ['Flight_Distance', 'Seat_comfort', 'Food_and_drink', 'Inflight_wifi_service', 
'Inflight_entertainment','Onboard_service', 'Leg_room_service', 'Baggage_handling',
'Cleanliness', 'Departure_Delay_in_Minutes', 'Arrival_Delay_in_Minutes']

train, test = train_test_split(q3, test_size=0.3, random_state=123)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state=123).fit(train[X_list], train.satisfaction)

from sklearn.metrics import f1_score

pred=lr.predict(test[X_list])
f1_score(test.satisfaction, pred, pos_label='dissatisfied')

# 답: 0.749





#%%

# =============================================================================
# =============================================================================
# # 문제 10 유형(DataSet_10.csv 이용)
#
# 구분자 : comma(“,”), 1538 Rows, 6 Columns, UTF-8 인코딩

# 중고 자동차 가격에 대한 분석을 위하여 아래와 같은 데이터를
# 확보하였다.

# 컬 럼 / 정 의 / Type
# model / 모델명 / String
# engine_power / 엔진 파워 / Double
# age_in_days / 운행 일수 / Double
# km / 운행 거리 / Double
# previous_owners / 이전 소유자 수 / Double
# price / 중고차 가격 / Double
# =============================================================================
# =============================================================================

# =============================================================================
# (참고)
# #1
# import pandas as pd
# import numpy as np
# #2
# import scipy.stats as ststs
# #3
# from sklearn.linear_model import LinearRegression
# =============================================================================

import pandas as pd
data10=pd.read_csv('Dataset_10.csv')
data10.info()

data10=data10.dropna(axis=1, how='all')
#%%

# =============================================================================
# 1.이전 소유자 수가 한 명이고 엔진 파워가 51인 차에 대해 모델별 하루 평균 운행
# 거리를 산출하였을 때 가장 낮은 값을 가진 모델이 가장 큰 값을 가진 모델에 대한
# 비율은 얼마인가? 소수점 셋째 자리에서 반올림하여 소수점 둘째 자리까지
# 기술하시오.
# (모델별 평균 → 일평균 → 최대최소 비율 계산) (답안 예시) 0.12
# =============================================================================

q1=data10.copy()

q1.columns

a=q1.previous_owners==1
b=q1.engine_power==51

q1[a & b].model.unique()
q1=q1[a & b]

q1['km_per_day']=q1['km']/q1['age_in_days']

l=q1[q1.model=='lounge']['km_per_day'].mean()
p=q1[q1.model=='pop']['km_per_day'].mean()
s=q1[q1.model=='sport']['km_per_day'].mean()

p/l

# ---------------------------
q1=data10.copy()

q1.columns

a=q1.previous_owners==1
b=q1.engine_power==51

q1[a & b].model.unique()
q1=q1[a & b]

q1_tab=pd.pivot_table(q1,index='model', values=['km', 'age_in_days'])

q1_tab['km_per_day'] = q1_tab['km'] / q1_tab['age_in_days']

q1_tab['km_per_day'].min()/q1_tab['km_per_day'].max()

# 답: 0.97
#%%

# =============================================================================
# 2.운행 일수에 대한 운행 거리를 산출하고, 위 1번 문제에서 가장 큰 값을 가지고 있던
# 모델과 가장 낮은 값을 가지고 있던 모델 간의 운행 일수 대비 운행거리 평균이 다른지
# 적절한 통계 검정을 수행하고 p-value를 소수점 세자리 이하는 버리고 소수점
# 두자리까지 기술하고 기각 여부를 Y / N로 답하시오. (등분산을 가정하고 equal_var = 
# True / var.equal = T로 분석을 실행하시오.)
# (답안 예시) 0.23, Y
# =============================================================================

# 가장 큰 값 모델: lounge
# 가장 낮은 값 모델: sport

q2=data10.copy()
q2['km_per_day']=q2['km']/q2['age_in_days']

group1=q2[q2.model=='lounge']['km_per_day']
group0=q2[q2.model=='sport']['km_per_day']

from scipy.stats import ttest_ind

t, p = ttest_ind(group0, group1, equal_var=True)
p

# 답: 0.13, N

#%%

# =============================================================================
# 3.독립변수로 engine_power, age_in_days, km를 사용하고 종속변수로 price를 사용하여
# 모델별 선형회귀분석을 수행하고, 산출된 모형을 사용하여 다음과 같은 조건의
# 중고차에 대한 가격을 예측하고 예측된 가격을 정수부만 기술하시오.
# - model : pop / engine_power : 51 / age_in_days : 400 / km : 9500 / previous_owners : 2

# (답안 예시) 12345
# =============================================================================
# model = pop이고 이전 소유자수가 2명인 데이터만을 이용하여 회귀모델을 생성하시오.

q3=data10.copy()

q3=q3[q3.model=='pop']
q3=q3[q3.previous_owners==2]

from statsmodels.formula.api import ols

ols10 = ols('price ~ engine_power + age_in_days + km', q3).fit()

ols10.predict({'engine_power':51, 'age_in_days':400, 'km':9500})

# 답: 10367











