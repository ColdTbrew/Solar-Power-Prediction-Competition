# 사용자 정의 손실 함수
import numpy as np
import pandas as pd
def custom_loss_function(y_true, y_pred):
    score = np.where(y_pred <= 15.84, 0, 
                     np.where(np.abs((y_pred - y_true) / 99 * 100) <= 6, 4 * y_true,
                              np.where(np.abs((y_pred - y_true) / 99 * 100) <= 8, 3 * y_true, 0)))
    return np.mean(score)