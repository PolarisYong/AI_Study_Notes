#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : linear_regression.py
@Time    : 2025/7/17 18:00
@Author  : Pan Yong
@Email   : panyong0417@gmail.com
@Desc    : linear regression with multiple features
"""
import os
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

# 2. 定义超参数
alpha = 0.01  # 学习率
num_iterations = 100  # 迭代次数


def compute_cost(X, y, theta):
    """计算损失函数J(theta)"""
    h = X @ theta  # 预测值，矩阵乘法
    cost_value = (1/(2*m)) * np.sum((h - y)**2)  # MSE损失函数
    return cost_value


def gradient_descent(X, y, theta, alpha, num_iterations):
    """梯度下降主函数"""
    cost_history = []  # 记录每次迭代的损失
    for i in range(num_iterations):
        h = X @ theta  # 预测值
        # 计算梯度（同时更新theta0和theta1）
        theta = theta - (alpha/m) * (X.T @ (h - y))
        # 记录损失
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history


def linear_regression(X, y, alpha, num_iterations):
    """
    线性回归模型，使用梯度下降法拟合数据。
    参数:
    X (numpy.ndarray): 特征矩阵，形状为 (m, n+1)，其中 m 是样本数，n 是特征数。
    y (numpy.ndarray): 目标值向量，形状为 (m,)。
    alpha (float): 学习率，控制梯度下降的步长。
    num_iterations (int): 迭代次数，控制梯度下降的次数。
    返回:
    theta (numpy.ndarray): 拟合后的参数向量，形状为 (n+1,)。
    """

    # 1. 数据预处理：添加x0=1
    X = np.c_[np.ones(m), x]  # 形状为(m, 2)，第一列全1
    # 2. 初始化参数
    theta = np.zeros(X.shape[1])  # [theta0, theta1]

    # 执行梯度下降
    return gradient_descent(X, y, theta, alpha, num_iterations)


if __name__ == '__main__':
    theta_final, cost_history = linear_regression(x, y, alpha, num_iterations)
    # 输出结果
    print(f"最终参数：theta0={theta_final[0]}, theta1={theta_final[1]}")

    # 可视化损失函数变化（验证收敛）
    plt.plot(cost_history)
    plt.xlabel("迭代次数")
    plt.ylabel("J(theta)")
    plt.title("损失函数随迭代的变化")
    try:
        plt.savefig('损失函数随迭代的变化.png', dpi=300, bbox_inches='tight')
        print("图片已保存至当前目录")
    except Exception as e:
        print(f"保存失败: {e}")
