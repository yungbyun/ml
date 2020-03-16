# 개인별 캐글 발표내용(*)
https://docs.google.com/spreadsheets/d/1vYmuk1hSTZ9j_y1AZEtA4EcAzU6ysljLmdXWZv_4xgI/edit?usp=sharing

# 캐글(kaggle.com) 데이터 과학 (붓꽃 Iris 인식)

캐글 홈페이지를 방문하여 Iris 검색, 이때 나오는 https://www.kaggle.com/uciml/iris 에서 다음 커널(kernels)을 찾아 선택 후 학습

## 데이터 놀이1 (3월 19일까지)
https://www.kaggle.com/ash316/ml-from-scratch-with-iris 

## 데이터 놀이2 (3월 22일까지)
https://www.kaggle.com/richbrosius/iris-classification-using-tensorflow

## 데이터 놀이3 (3월 26일까지)
https://www.kaggle.com/lavajiit/deep-learning-iris-dataset-keras

# 우리는 왜 대학에 가는가? (동영상)
https://www.youtube.com/watch?v=nttlAfVQT6w


# 가장 간단한 코드(케라스)
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

