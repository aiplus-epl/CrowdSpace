import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

file_name = ''
data = pd.read_csv(file_name)

# RMSprop 옵티마이저 설정을 위한 함수
def myOptimizer(lr):          # lr: learning rate
    return tf.keras.optimizers.RMSprop(learning_rate=lr)

# Discriminator와 Generator에 사용될 모델 파라미터들을 설정
d_input = data.shape[1]      # 데이터의 피처 수를 입력값으로 설정
d_hidden = 256               # Discriminator의 은닉층 노드 수
d_output = 1                 # Discriminator의 출력 노드 수
g_input = 32                 # Generator의 입력 노드 수
g_hidden = 256               # Generator의 은닉층 노드 수
g_output = d_input           # Generator의 출력 노드 수 (Discriminator의 입력과 동일하게 설정)

# -1에서 1 사이의 무작위 값을 가진 배열(z)를 생성하는 함수
def makeZ(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])

# Discriminator 모델을 생성하는 함수
def build_D():
    d_x = tf.keras.layers.Input(shape=(d_input,))              # 입력층
    d_h = tf.keras.layers.Dense(d_hidden, activation='relu')(d_x)  # 은닉층
    d_o = tf.keras.layers.Dense(d_output, activation='sigmoid')(d_h)  # 출력층

    d_model = tf.keras.models.Model(d_x, d_o)  # 모델을 구성
    d_model.compile(loss='binary_crossentropy', optimizer=myOptimizer(0.0001))  # 모델을 컴파일

    return d_model

# Generator 모델을 생성하는 함수
def build_G():
    g_x = tf.keras.layers.Input(shape=(g_input,))                 # 입력층
    g_h = tf.keras.layers.Dense(g_hidden, activation='relu')(g_x)    # 은닉층
    g_o = tf.keras.layers.Dense(g_output, activation='linear')(g_h)  # 출력층

    g_model = tf.keras.models.Model(g_x, g_o)  # 모델을 구성
    return g_model

# GAN 모델을 생성하는 함수. Discriminator와 Generator를 결합
def build_GAN(discriminator, generator):
    discriminator.trainable = False    # GAN 훈련시 판별자는 훈련되지 않도록 설정
    z = tf.keras.layers.Input(shape=(g_input,))
    Gz = generator(z)  # 생성자를 통해 가짜 데이터를 생성
    DGz = discriminator(Gz)  # 판별자를 통해 가짜 데이터를 평가

    gan_model = tf.keras.models.Model(z, DGz)
    gan_model.compile(loss='binary_crossentropy', optimizer=myOptimizer(0.0005))

    return gan_model

# 현재 텐서플로우 세션의 모든 변수와 연산을 초기화
tf.keras.backend.clear_session()

# 각 모델을 생성합니다.
D = build_D()   # Discriminator
G = build_G()   # Generator
GAN = build_GAN(D, G)  # GAN

n_batch_cnt = 6
n_batch_size = int(data.shape[0] / n_batch_cnt)
EPOCHS = 10

# 주어진 epoch 수만큼 GAN을 학습
for epoch in range(EPOCHS):
    for n in range(n_batch_cnt):
        # 미니 배치의 시작과 끝 인덱스를 설정
        from_, to_ = n * n_batch_size, (n + 1) * n_batch_size
        if n == n_batch_cnt - 1:
            to_ = data.shape[0]

        # 실제 데이터의 미니 배치를 가져옴
        X_batch = data[from_:to_]
        # Generator에 넣을 무작위 노이즈를 생성
        Z_batch = makeZ(m=X_batch.shape[0], n=g_input)
        # Generator를 사용하여 가짜 데이터를 생성
        Gz = G.predict(Z_batch)

        # Discriminator를 학습하기 위한 목표 값들을 설정
        d_target = np.zeros(X_batch.shape[0] * 2)
        d_target[:X_batch.shape[0]] = 0.9  # 진짜 데이터에 대한 목표 값
        d_target[X_batch.shape[0]:] = 0.1  # 가짜 데이터에 대한 목표 값

        # 실제 데이터와 가짜 데이터를 결합
        bX_Gz = np.concatenate([X_batch, Gz])

        # Generator를 학습하기 위한 목표 값들을 설정
        g_target = np.zeros(Z_batch.shape[0])
        g_target[:] = 0.9

        # Discriminator와 Generator를 각각 학습시킴
        loss_D = D.train_on_batch(bX_Gz, d_target)
        loss_G = GAN.train_on_batch(Z_batch, g_target)

    # 10 epoch마다 학습 상황을 출력
    if epoch % 10 == 0:
        z = makeZ(m=data.shape[0], n=g_input)
        print("Epoch: %d, D-loss = %.4f, G-loss = %.4f" % (epoch, loss_D, loss_G))

# 학습 후 Generator를 사용하여 가짜 데이터를 생성하고 결과를 저장
z = makeZ(m=data.shape[0], n=g_input)
fake_data = G.predict(z)
result = pd.DataFrame(fake_data)
print(result)
result.to_csv('fake_data_GAN.csv')
