import numpy as np
import tensorflow as tf

# 다중 시계열 예측에 사용할 기본 메트릭 클래스
class MultiHorizonMetricTF:
    def to_prediction(self, y_pred):
        # 예측값을 반환하는 메서드
        return y_pred

# 평균 제곱 오차(MSE) 손실 클래스
class myMSETF(MultiHorizonMetricTF):
    def loss(self, y_pred, y_true):
        # 손실 계산 메서드
        y_pred = self.to_prediction(y_pred)
        loss = tf.square(y_true - y_pred)
        return loss

# 주관사 정의 가중치가 추가된 평균 제곱 오차 손실 클래스
class myMSEPLUSTF(MultiHorizonMetricTF):
    def __init__(self, alpha1=100, alpha2=100):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def loss(self, y_pred, y_true):
        y_pred = self.to_prediction(y_pred)
        loss = y_true - y_pred

        # 조건에 따라 가중치 적용
        weight = tf.where(loss > 0, 1, self.alpha1)
        weight2 = tf.where(y_true * y_pred > 0, 1, self.alpha2)

        # 가중치와 함께 손실 계산
        weight_loss = tf.square(loss) * weight * weight2
        return weight_loss

# 주관사 정의 가중치가 추가된 평균 제곱 오차 손실 클래스 (다른 조건)
class myMSEMINUSTF(MultiHorizonMetricTF):
    def __init__(self, alpha1=100, alpha2=100):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def loss(self, y_pred, y_true):
        y_pred = self.to_prediction(y_pred)
        loss = y_true - y_pred

        # 조건에 따라 가중치 적용
        weight = tf.where(loss > 0, self.alpha1, 1)
        weight2 = tf.where(y_true * y_pred > 0, 1, self.alpha2)

        # 가중치와 함께 손실 계산
        weight_loss = tf.square(loss) * weight * weight2
        return weight_loss

# 평균 절대 백분율 오차(MAPE) 손실 클래스
class myMAPETF(MultiHorizonMetricTF):
    def loss(self, y_pred, y_true, eps=1e-10):
        y_pred = self.to_prediction(y_pred)
        mape = tf.abs(y_true - y_pred) / (tf.abs(y_true) + eps)
        return mape
