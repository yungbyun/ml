# 매일 설문 (강의 시작 바로 전에)
https://docs.google.com/forms/d/e/1FAIpQLSdpNqMkiQLPf9_uGq7Jel2kxINxL-RNEGO9lHIxZlTXSCSxcA/viewform?usp=pp_url

# 캐글 해커톤 신청 (다음 주 금요일 저녁부터 토요일 오전까지)
https://docs.google.com/forms/d/e/1FAIpQLSfVmhbxyi4IPaI3tF4asd7b7OWDJpiwdGtTnqEtqhTCrCymLg/viewform

# 개인별 캐글 발표내용(*)
https://docs.google.com/spreadsheets/d/1vYmuk1hSTZ9j_y1AZEtA4EcAzU6ysljLmdXWZv_4xgI/edit?usp=sharing

# 거꾸로 학습 후 이해한 내용과 궁금한 내용 
https://goo.gl/forms/e9ZReC0471dA4bCN2

# 캐글 튜토리얼 참가신청
3월 22일 첨단과학단지(ICT융합창업허브 2층) 

https://docs.google.com/forms/d/e/1FAIpQLSdKbwmUAPgPGzhO2RY-sE48cyY1IDJzusf8fD0dkfRi4DE8Sg/viewform

# 캐글(kaggle.com) 데이터 과학 (붓꽃 Iris 인식)

캐글 홈페이지를 방문하여 Iris 검색, 이때 나오는 https://www.kaggle.com/uciml/iris 에서 다음 커널(kernels)을 찾아 선택 후 학습

## 데이터 놀이1 (3월 19일까지)
https://www.kaggle.com/ash316/ml-from-scratch-with-iris 

## 데이터 놀이2 (3월 22일까지)
https://www.kaggle.com/richbrosius/iris-classification-using-tensorflow

## 데이터 놀이3 (3월 26일까지)
https://www.kaggle.com/lavajiit/deep-learning-iris-dataset-keras


# 제2회 캐글코리아 경진대회 참가
https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr


# 우리는 왜 대학에 가는가? (동영상)
https://www.youtube.com/watch?v=nttlAfVQT6w


from keras.models import Sequential
from keras.layers import Dense

x_data = [1]
y_data = [1]

gildong = Sequential()

l = Dense(1, activation='linear', input_dim=1)
gildong.add(l)

gildong.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

gildong.fit(x_data,y_data,epochs =200,batch_size = 32)

answer = gildong.predict(x_data)
print('Predicted:', answer)


