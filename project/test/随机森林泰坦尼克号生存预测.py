import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 导入数据
df = pd.read_csv("./data/train.csv")

# 数据预处理（找到特征和标签）
data = df[["Pclass", "Age", "Sex"]]
target = df["Survived"]

# 对Age进行均值填充
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 对Sex进行独热编码
data = pd.get_dummies(data, columns=["Sex"], drop_first=True)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# 特征工程
ss = StandardScaler()
x_train = ss.fit_transform(X_train)
x_test = ss.fit_transform(X_test)

# 设置参数组合
param_grid = {
    'n_estimators': [i for i in range(7, 30, 2)],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5]
}

# 模型选择
model = RandomForestClassifier(random_state=42)

# 交叉网格验证
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5
)

grid_search.fit(x_train, y_train)

# 输出最优参数和最优交叉验证分数
print("+++++++++++++++++++++++++++++++++++++++++++++++")
print("网格搜索结果：", grid_search.best_params_, grid_search.best_score_)
print("+++++++++++++++++++++++++++++++++++++++++++++++")

# 使用最优网格模型训练参数
best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)
print("准确率：", best_model.score(x_test, y_test))
print("预测结果为：", best_model.predict(x_test))
print("准确率：", accuracy_score(y_test, best_model.predict(x_test)))
print("精确率：", precision_score(y_test, best_model.predict(x_test)))
print("召回率：", recall_score(y_test, best_model.predict(x_test)))
print("F1值：", f1_score(y_test, best_model.predict(x_test)))



