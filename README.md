# 머신러닝(Machine Learning)

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

