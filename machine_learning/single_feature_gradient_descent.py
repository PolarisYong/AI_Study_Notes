#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : single_feature_gradient_descent.py
@Time    : 2025/7/17 16:44
@Author  : Pan Yong
@Email   : panyong0417@gmail.com
@Desc    : This Python script implements gradient descent for single-feature linear regression,
            aiming to minimize the mean squared error (MSE) between predicted and actual values.
"""
import numpy as np
import matplotlib.pyplot as plt


# 指定字体，这里以 SimSun 为例（Windows 系统常见，其他系统可替换成对应支持中文的字体）
plt.rcParams['font.sans-serif'] = ['SimSun']
# 解决负号显示为方块的问题（可选，若有负号显示异常时使用）
plt.rcParams['axes.unicode_minus'] = False


# 1. 准备数据（假设特征x和目标y）
x = np.array([1, 2, 3, 4, 5], dtype=np.float64)  # 特征
y = np.array([2, 4, 5, 4, 5], dtype=np.float64)  # 目标值
m = len(y)  # 样本数

# 2. 数据预处理：添加x0=1
X = np.c_[np.ones(m), x]  # 形状为(m, 2)，第一列全1

# 3. 初始化参数
theta = np.zeros(2)  # [theta0, theta1]

# 4. 定义超参数
alpha = 0.01  # 学习率
num_iterations = 1000  # 迭代次数


# 5. 梯度下降迭代
def compute_cost(X, y, theta):
    """计算损失函数J(theta)"""
    m = len(y)
    h = X @ theta  # 预测值，矩阵乘法
    J = (1/(2*m)) * np.sum((h - y)**2)  # MSE损失函数
    return J


def gradient_descent(X, y, theta, alpha, num_iterations):
    """梯度下降主函数"""
    m = len(y)
    J_history = []  # 记录每次迭代的损失
    for i in range(num_iterations):
        h = X @ theta  # 预测值
        # 计算梯度（同时更新theta0和theta1）
        theta = theta - (alpha/m) * (X.T @ (h - y))
        # 记录损失
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history


# 执行梯度下降
theta_final, J_history = gradient_descent(X, y, theta, alpha, num_iterations)

# 输出结果
print(f"最终参数：theta0={theta_final[0]}, theta1={theta_final[1]}")

# 可视化损失函数变化（验证收敛）
plt.plot(J_history)
plt.xlabel("迭代次数")
plt.ylabel("J(theta)")
plt.title("损失函数随迭代的变化")
plt.savefig('损失函数随迭代的变化')