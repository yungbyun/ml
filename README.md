## 제주RIS 지능형서비스 사업단 프로그램(홍보)
> https://www.youtube.com/watch?v=5ny7G_39hbQ&t=22s

## (4학년) 중기청 200만원 장학금 지원 및 취업지원사업 (참여신청중! 선착순!)
> * https://github.com/JNUAca/go
> * 참여신청: https://forms.gle/qnfsxRzCAfwAG4gi8

## 취업 및 행사정보: 바울랩 카카오 오픈 단톡방
> * https://open.kakao.com/o/gKG3e0sb

## Kaggle 과제 정보 입력(결정트리와 같은 일반 머신러닝 모델이 아닌 신경망을 이용한 딥러닝 모델 코드 분석)
> * https://docs.google.com/spreadsheets/d/18InS788sBL-napveBnIvh65Rf9GGpslL5rncM98bCm0/edit#gid=0
> * 제출일자: TBA (yungcheolbyun@gmail.com으로 제출)
> * 제출형식: 발표자료 형식(PPT): 데이터 및 시각화, 데이터 전처리, 딥러닝 모델, 학습 및 테스트 결과, 결론 등

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
```python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 데이터
X = [1, 2, 3]
y = [1, 2, 3]

# 1개의 신경세포
gildong = Sequential()
neuron = Dense(1, input_dim=1, activation='linear')
gildong.add(neuron)

# 모델 컴파일
gildong.compile(optimizer=Adam(learning_rate=0.02), loss='mse')

# 학습
gildong.fit(X, y, epochs=1000, verbose=0)

# 예측/테스트
answer = gildong.predict(X)
print(f"Prediction: {answer}")

```

## CNN
* https://www.kaggle.com/code/pavansanagapati/a-simple-cnn-model-beginner-guide
  
## 강의 동영상 리스트 (예습복습 필수!)
* 실습(성별예측1): https://youtu.be/QBq2f_1gfZA 
* 실습(성별예측2): https://youtu.be/4IEbdh62d2Y
* 실습(식물생장예측): https://youtu.be/DZnpkKxeB-w
 
* 논리회귀1: https://youtu.be/vztm69wYlhs
* 논리회귀2, https://youtu.be/gmEOiuB-TKM
* 논리회귀3, https://youtu.be/greSudDm4Gc
* 논리회귀4, https://youtu.be/z777NFDLNkQ
* 논리회귀5, https://youtu.be/Tu9iyy4RaVo
* 논리회귀6, https://youtu.be/Rl5bia3ts1U
* 논리회귀7, https://youtu.be/LI6whxHBNT4
* 논리회귀8, https://youtu.be/hqkNSTwrK9M
* 논리회귀9, https://youtu.be/lhUFXIGhPEs
* 논리회귀10, https://youtu.be/p6lLyh0G8RQ
* 논리회귀11, https://youtu.be/IKpSvmsaDew
* 논리회귀12, https://youtu.be/85hMwhKDSw4
* 논리회귀13, https://youtu.be/2sAg8ze7K_U
* 논리회귀14, https://youtu.be/GT9s4_f22TU
* 논리회귀15, https://youtu.be/JGAjfgAGFkQ
* 딥러닝1, https://youtu.be/4rjma0B5Z3A
* 딥러닝2, https://youtu.be/PVqInsNHgJM
* 딥러닝3, https://youtu.be/QhM9IOHJRwc
* 딥러닝4, https://youtu.be/t6HumFBBQ74

## 비주얼 스튜디오 코드 설치 및 간단 사용법
> * https://www.youtube.com/watch?v=K8qVH8V0VvY&t=331s

## 비주얼 스튜디오 코드를 이용한 버전관리 (git 사용법)
> * [Visual Studio Code에서 Git으로 협업하기1](https://www.youtube.com/watch?v=vI8FFvQge2w)
> * [Visual Studio Code에서 Git으로 협업하기2](https://www.youtube.com/watch?v=m1Q9dXuD-O0)
> * [Visual Studio Code에서 Git으로 협업하기3](https://www.youtube.com/watch?v=srN5UchM26A&t=27s)
> * [Visual Studio Code에서 Git으로 협업하기4](https://www.youtube.com/watch?v=GfUj_ZvRppg&t=20s)
> * [Visual Studio Code에서 Git으로 협업하기5](https://www.youtube.com/watch?v=AEhPmgDdzYg&list=PLuHgQVnccGMAQvSVKdXFiOo51HUD8iQQm&index=5)
> * [Visual Studio Code에서 Git으로 협업하기6](https://www.youtube.com/watch?v=P-UNibp1FHg&list=PLuHgQVnccGMAQvSVKdXFiOo51HUD8iQQm&index=6)
> * [Visual Studio Code에서 Git으로 협업하기7](https://www.youtube.com/watch?v=yc0rxkt93MQ&list=PLuHgQVnccGMAQvSVKdXFiOo51HUD8iQQm&index=7)
