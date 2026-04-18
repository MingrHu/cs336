import numpy as np
from typing import Any

def r2_score(y_true: np.ndarray, y_pred: np.ndarray)->float:
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def create_regression_data(samples=1000, input_dim=5, output_dim=1)->Any:
    # 生成输入 X：随机数据
    X = np.random.randn(samples, input_dim)
    
    # 生成真实 Y：用一个线性关系 + 噪声（适合回归）
    true_w = np.array([[2.5], [-1.3], [0.8], [3.1], [-0.5]])
    Y = X @ true_w + 0.1 * np.random.randn(samples, output_dim)  # 加噪声
    
    return X, Y

def std_scaler_mat(mat:np.ndarray)->Any:
    mat_mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    scaled_mat = (mat - mat_mean) / std
    return scaled_mat,std,mat_mean

def re_std_scaler(scaled_mat:np.ndarray,std:np.ndarray,mat_mean:np.ndarray)->np.ndarray:
    return scaled_mat * std + mat_mean