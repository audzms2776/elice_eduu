import numpy as np

# train Parameters
seq_length = 7

# Open, High, Low, Volume, Close
xy = np.loadtxt('stock.csv', delimiter=',')
# xy = xy[::-1]  # reverse order (chronically ordered)

global_min = np.min(xy, 0)
numerator = xy - global_min
# 최대 - 최소 계산
denominator = np.max(xy, 0) - np.min(xy, 0)
xy = numerator / (denominator + 1e-7)

x = xy  # 데이터 전부
y = xy[:, [-1]]  # Close as label 종가 데이터

last_data = x[-seq_length:]
last_pred = y[-seq_length:]

# print(global_min)
# print(last_data * (denominator + 1e-7) + global_min)
print()
