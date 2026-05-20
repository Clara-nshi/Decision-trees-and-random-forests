# 机器学习库
import inline
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, classification_report
import matplotlib
matplotlib.use('TkAgg')

# XGBoost
import xgboost as xgb
from sympy.plotting.backends.matplotlibbackend import matplotlib

# 设置风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 加载乳腺癌数据集（经典二分类）
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练决策树
dt_clf = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
dt_clf.fit(X_train, y_train)

# 评估
y_pred = dt_clf.predict(X_test)
print(f"决策树测试集准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"决策树测试集AUC: {roc_auc_score(y_test, dt_clf.predict_proba(X_test)[:,1]):.4f}")

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dt_clf,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Visualization", fontsize=16)
plt.show()