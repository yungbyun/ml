# 온라인 강의실(Slack)
> * https://join.slack.com/t/w1614659986-7xj229671/shared_invite/zt-n3cxmu37-fftbMTur7t31AX724AA99g

# 캐글(kaggle.com)
> * 가입 및 둘러보기 (Slack 01주차_2 채널 참고)

# 캐글 코드 다운받아 실행해보기
> * 캐글 홈페이지를 방문하여 Iris 검색해보자. 그리고 아래의 검색되는 코드를 다운받은 후(Copy and Edit) 실행해보자!
> ## 데이터 놀이1 (3월 16일까지)
> * https://www.kaggle.com/ash316/ml-from-scratch-with-iris 
> ## 데이터 놀이2 (3월 23일까지)
> * https://www.kaggle.com/richbrosius/iris-classification-using-tensorflow
> ## 데이터 놀이3 (3월 30일까지)
> * https://www.kaggle.com/lavajiit/deep-learning-iris-dataset-keras

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

