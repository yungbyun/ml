# 머신러닝(Machine Learning)

## 퀴즈(평가반영) 참여자
> * 04월 12일: 정답 오버슈팅(발산), 김진웅, 김경범, 안상민, 김희범, 전찬혁, 김승현, 권영기, 정승원 (제출순서) 
> * 05월 04일: 시그모이드 함수에 대하여: 허승범, 권영기, 정승원, 변세민, 백예림, 김승헌 

## 딥러닝 실습코드
> * (뉴런과 신경망 학습 코드) https://github.com/yungbyun/myml -> 구글 코랩(Google Colab)에서 실행하세요.

## 머신러닝 실습코드
> * (성별 인식 코드) https://www.kaggle.com/yungbyun/female-male-classification-ml-simple/edit/run/30600474 -> 캐글에서 바로 실행하세요.
> * (식물생장) https://www.kaggle.com/code/yungbyun/plant-diary-original-simple -> 캐글에서 바로 실행하세요.

## 참고용 캐글 실습 코드
> 캐글 홈페이지를 방문하여 Iris 검색해보자. 그리고 아래의 검색되는 코드를 다운받은 후(Copy and Edit) 실행해보세요.
> ### 데이터 놀이1 
> * https://www.kaggle.com/ash316/ml-from-scratch-with-iris 
> ### 데이터 놀이2 
> * https://www.kaggle.com/richbrosius/iris-classification-using-tensorflow
> ### 데이터 놀이3 
> * https://www.kaggle.com/akashsri99/deep-learning-iris-dataset-keras

## 가장 간단한 코드(케라스)
```csharp
from keras.models import Sequential
from keras.layers import Dense

x_data = [1]
y_data = [1]

gildong = Sequential()

l = Dense(1, activation='linear', input_dim=1)
gildong.add(l)

gildong.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

gildong.fit(x_data, y_data, epochs = 200)

answer = gildong.predict(x_data)
print('Predicted:', answer)
```

