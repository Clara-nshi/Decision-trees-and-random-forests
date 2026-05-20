# 导包
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 准备数据
df = pd.read_csv("./data/train.csv")
print(df.shape)

# 2. 数据预处理
# 先分别获取特征和标签
x = df[["Pclass", "Age", "Sex"]]
y = df["Survived"]

# 缺失值处理填充年龄平均值
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 提前把性别转成独热编码
x = pd.get_dummies(x, columns=["Sex"], drop_first=True)
print(x)

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)


# 4. 标准化
# ss = StandardScaler()
# x_train = ss.fit_transform(X_train)
# x_test = ss.transform(X_test)

# todo 5. 模型选择
model = DecisionTreeClassifier(criterion="gini")    # 默认CART分类

# 6. 模型训练
model.fit(X_train, y_train)

# 7. 模型预测
y_predict = model.predict(X_test)
print("预测结果为：", y_predict)
print("调用API计算准确率：", accuracy_score(y_test, y_predict))
print("直接计算准确率：", model.score(X_test, y_test))

# 8. 绘制决策树
# 设置画布大小
plt.figure(figsize=(50, 40))
# 绘制决策树， 参数1：模型， 参数2：是否填充颜色
plot_tree(model, filled=True)
# 保存图片
plt.savefig("./data/titanic_tree.png")
# 展示
plt.show()