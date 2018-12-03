# 사전 설문
https://docs.google.com/forms/d/1b8tymxNydEIKLTrqzZ3rsUIM75eZUrD-Ejkpi5DVKqw/edit

''' python
import tensorflow as tf

x_data = [1]
y_data = [1]

#----- 신경세포 만들기
w = tf.Variable(tf.random_normal([1]))
hypo = w * x_data

#----- 신경세포 학습시키기
cost = (hypo - y_data) ** 2
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_list = []
print('w:', sess.run(w), 'cost:', sess.run(cost))
for i in range(1001) :
    sess.run(train)
    if i % 100 == 0:
        err_val = sess.run(cost)
        print('w:', sess.run(w), 'cost:', err_val)
        cost_list.append(err_val)

# 오류가 줄어드는 모습 보기
import matplotlib.pyplot as plt
plt.plot(cost_list)
plt.show();

#----- 테스트/예측
print(sess.run(w * [3]))
''' 
