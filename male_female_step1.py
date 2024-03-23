# 키, 몸무게, 발 크기로 성별 알아맞추기

import pandas as pd # 데이터 처리를 위한 모듈
import seaborn as sns # 데이터 시각화 모듈
import matplotlib.pyplot as plt # 데이터 시각화 모듈



# 데이터를 불러와서 분석하기 위해 pandas 모듈을 사용한다. <br/>

# 1. 데이터 불러오고 정보 표시
gildong = pd.read_csv("iris.csv")
print(gildong.head(5))
gildong.info()

# 2. 데이터 전처리
# Id라는 컬럼(axis=1) 삭제(drop) 후 데이터 프레임에 반영하라(inplace=True).
gildong.drop('Id',axis=1,inplace=True)
print(gildong.head(5))

# 3. 데이터 시각화 및 분석
# 특징에 따른 성별 분포를 플로팅하여 보자. <br/>
def myplot(a, b, c):
    fig = gildong[gildong[c]==0].plot(kind='scatter',x=a,y=b,color='orange', label='Female')
    gildong[gildong[c]==1].plot(kind='scatter',x=a,y=b,color='blue', label='Male', ax=fig)
    fig.set_xlabel(a)
    fig.set_ylabel(b)
    fig.set_title(a + " vs. " + b)
    fig=plt.gcf()
    fig.set_size_inches(10,6)
    plt.show()

myplot("FeetSize", "Height", "Sex")
myplot("Height", "FeetSize", "Sex")

# 각 특징(키/몸무게/발크기/학년/성별)별로 어떤 분포를 보이는지도 표시할 수 있다.
def myhist(a):
    a.hist(edgecolor='black', linewidth=1.2)
    fig = plt.gcf()
    fig.set_size_inches(12,10)
    plt.show()

myhist(gildong)

# > 어떻게 분포하는지도 알 수 있다. 바이올린 모양으로 표시할 수 있는데 violinplot 함수를 이용한다. <br/>
# > 성별에 따라 꽃받침 너비, 꽃받침 길이, 꽃잎 너비, 꽃잎 길이 등이 어떻게 분포하는지 알 수 있다. <br/>
def myviolinplot(df, a, b):
    plt.figure(figsize=(5,4))
    plt.subplot(1,1,1)
    sns.violinplot(x=a,y=b,data=df)

myviolinplot(gildong, 'Sex', 'Height')
#myviolinplot(gildong, 'Sex', 'Weight')
#myviolinplot(gildong, 'Sex', 'FeetSize')
#myviolinplot(gildong, 'Sex', 'Year')

# 다양한 분류 알고리즘 패키지를 임포트
from sklearn.linear_model import LogisticRegression  #Logistic Regression
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.neighbors import KNeighborsClassifier #K-nearest neighbours
from sklearn import svm  #Support Vector Machine (SVM) Algorithm
from sklearn import metrics #checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #Decision Tree Algoithm

print(gildong.shape) # 데이터 모양(shape)을 표시
print(gildong.head(5))

# 4. 특징 중요도 시각화
def display_heatmap(df):
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(),annot=True,cmap='cubehelix_r')
    plt.show()

display_heatmap(gildong)

# 머신러닝 구현 절차는 다음과 같다.
#  1. 데이터 쪼개기 (학습용, 테스트용) <br/>
#  2. 알고리즘 선택 (학습시킬 아기 객체 만들기) <br/>
#  3. 아기의 **.fit()** 함수 호출하여 학습 시키키 <br/>
#  4. 아기의 **.predict()** 함수를 호출하여 테스트하기 <br/>
#  5. 아기가 예측한 값과 실제 정답 비교하여 인식률 계산하기

# 5. 데이터 쪼개기 (학습 데이터, 테스트 데이터)
train, test = train_test_split(gildong, test_size = 0.3)
# train=70% and test=30%
print(train.shape)
print(test.shape)

# 6. 학습/테스트
# 6.1 학습/테스트 데이터 구성
train_X = train[['Height','Weight','FeetSize']] # 키와 발크기만 선택
train_y = train.Sex # 정답 선택

test_X = test[['Height','Weight','FeetSize']] # taking test data features
test_y = test.Sex   #output value of test data

print(test_X)
print(test_y)

# 6.2 써포트 벡터 머신(SVM) 알고리즘 이용하여 알아맞추기
baby1 = svm.SVC() # 애기
baby1.fit(train_X,train_y) # 가르친 후
prediction = baby1.predict(test_X) # 테스트
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)

# 6.3 논리 회귀(Logistic Regression) 알고리즘 이용하여 알아맞추기
baby2 = LogisticRegression()
baby2.fit(train_X,train_y)
prediction = baby2.predict(test_X)
print('인식률:', metrics.accuracy_score(prediction,test_y) * 100)

# 6.4 결정 트리(Decision Tree) 알고리즘 이용하여 알아맞추기
baby3 = DecisionTreeClassifier()
baby3.fit(train_X,train_y)
prediction = baby3.predict(test_X)
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)

# 6.5 근접 이웃(K-Nearest Neighbours) 알고리즘 이용하여 알아맞추기
baby4 = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
baby4.fit(train_X,train_y)
prediction = baby4.predict(test_X)
print('인식률:',metrics.accuracy_score(prediction,test_y) * 100)
