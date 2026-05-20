import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 读取数据
df = pd.read_csv("./data/train.csv")

# 数据预处理
# 提取特征和标签
x = df[['Pclass', 'Age', 'Sex']].copy()
y = df['Survived']

# 对缺失值进行处理
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 对性别进行独热编码
x = pd.get_dummies(x, columns=['Sex'], drop_first=True)

# 数据划分为测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 特征工程
ss = StandardScaler()
x_train = ss.fit_transform(X_train)
x_test = ss.transform(X_test)

# 选择模型
model = DecisionTreeClassifier(criterion="gini")
# 模型训练
model.fit(x_train, y_train)
# 模型预测
model_pred = model.predict(x_test)
print("预测结果：", model_pred)
# 模型评估
print("预测准确率", accuracy_score(y_test, model_pred))
print("预测精确率", precision_score(y_test, model_pred))
print("预测召回率", recall_score(y_test, model_pred))

# 绘制图像
plt.figure(figsize=(60, 70))
plot_tree(model, filled=True)
plt.savefig("./data/titanic_tree")
plt.show()