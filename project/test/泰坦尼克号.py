import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# 数据加载
df = pd.read_csv("./data/train.csv")

# 数据预处理
x = df[['Pclass', 'Age', 'Sex']]
y = df['Survived']

# 年龄的缺失值用年龄的平均值填充
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 性别转换成独热编码
x = pd.get_dummies(x, columns=['Sex'], drop_first=True)
print(x)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# 标准化
ss = StandardScaler()
x_train = ss.fit_transform(X_train)
x_test = ss.transform(X_test)

# 模型选择
model = DecisionTreeClassifier(criterion="gini")
# 模型训练
model.fit(x_train, y_train)
# 模型预测
y_model = model.predict(x_test)
print("模型预测结果为：", y_model)
# 模型评估
print("准确率：", accuracy_score(y_test, model.predict(x_test)))
print("精确率：", precision_score(y_test, model.predict(x_test)))
print("召回率：", recall_score(y_test, model.predict(x_test)))


