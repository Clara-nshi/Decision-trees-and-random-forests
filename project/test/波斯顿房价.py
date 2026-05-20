# 导包
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

# 导入数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# 数据预处理， 提取 data 特征和 target 标签
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 数据切割
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 特征工程
ss = StandardScaler()
x_train = ss.fit_transform(X_train)
x_test = ss.transform(X_test)

# 模型选择
model = DecisionTreeRegressor(criterion="squared_error")
# 模型训练
model.fit(x_train, y_train)

# 模型预测
y_predict = model.predict(x_test)
print("预测房价为：", y_predict)

# 模型评估
print("均方误差：", mean_squared_error(y_test, y_predict))
print("均方根误差：", root_mean_squared_error(y_test, y_predict))
print("平均绝对误差：", mean_absolute_error(y_test, y_predict))