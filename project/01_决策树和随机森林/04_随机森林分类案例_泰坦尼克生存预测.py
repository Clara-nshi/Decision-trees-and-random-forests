# 导包
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 读取数据
df = pd.read_csv("./data/train.csv")


# 2. 数据预处理
x = df[['Pclass', 'Age', 'Sex']]
y = df['Survived']

# 缺失值处理
x = x.copy()
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 性别转换成独热编码
x = pd.get_dummies(x, columns=['Sex'], drop_first=True)

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 4. 特征工程
ss = StandardScaler()
x_train = ss.fit_transform(X_train)
x_test = ss.transform(X_test)

# 定义要搜索的参数组合
param_grid = {
    'n_estimators': [i for i in range(7, 30, 2)],     # 树的数量
    'max_depth': [3, 5, 7],             # 每棵树最大深度
    'min_samples_split': [2, 5]         # 内部节点再划分所需最小样本数
}

# 5. 模型选择
model = RandomForestClassifier(random_state=42)

# 网格搜索 + 交叉验证
grid_search = GridSearchCV(
    estimator=model,            # 模型
    param_grid=param_grid,      # 参数范围
    scoring='accuracy',         # 5折交叉验证
    cv=5                        # 优化目标：准确率
)

# 训练网格搜索
grid_search.fit(x_train, y_train)

# 输出最优参数和最优交叉验证分数
print("+++++++++++++++++++++++++++++++++++++++++++++++")
print("网格搜索结果：", grid_search.best_params_, grid_search.best_score_)
print("+++++++++++++++++++++++++++++++++++++++++++++++")

# 使用最优参数训练最终模型
best_model = grid_search.best_estimator_


# 交叉验证评估
cv_scores = cross_val_score(
    estimator=best_model,
    X=x_train,
    y=y_train,
    scoring='accuracy',
    cv=5
)
print("交叉验证结果：", cv_scores)
print("交叉验证平均结果：", cv_scores.mean())
print("+++++++++++++++++++++++++++++++++++++++++++++++")

# 7. 模型预测
y_predict = best_model.predict(x_test)
print("预测结果为：", y_predict)
# 8. 模型评估
print("准确率：", best_model.score(x_test, y_test))
print("准确率：", accuracy_score(y_test, y_predict))
print("精确率：", precision_score(y_test, y_predict))
print("召回率：", recall_score(y_test, y_predict))
