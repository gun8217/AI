from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt

# 데이터 준비
file_path = 'C:/Users/Administrator/Documents/Git/AI/ML/RandomForest/iris/data/Iris.csv'
iris = pd.read_csv(file_path, index_col=0)
# print(iris.head())

X = iris[iris.columns[:-1]]
y = iris[iris.columns[-1]]

# 데이터 분할(훈련과 테스트 세트로 구성)
# 훈련 = 데이터 패턴 학습 및 가중치 조정
# 테스트 = 일반화 성능 평가(무작위 데이터에서 얼마나 잘 작동하는지)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=42)

# 모델 훈련(knn, gusiann, decition 등의 모델을 바꿔 사용 가능?!)
decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=32)
decision_tree.fit(X_train, y_train)

# 모델 평가
train_acc = decision_tree.score(X_train, y_train)
print("훈련 정확도 :", train_acc)

pred = decision_tree.predict(X_test) # 모델 테스트

test_acc = decision_tree.score(X_test, y_test)
print("테스트 정확도 :", test_acc)

cm = confusion_matrix(y_test, pred)
print(f"예측 결과 확인: {cm}")



for attr in dir(decision_tree):
    if not attr.startswith('__'):print(decision_tree)
    break
    
print(decision_tree.criterion)
print(decision_tree.feature_names_in_)
print(decision_tree.feature_importances_)



from sklearn import tree

plt.figure(figsize=(5, 10), dpi=300)
tree.plot_tree(decision_tree,
               feature_names=iris.columns[:-1],
               impurity=True, filled=True,
               rounded=True)

plt.savefig('./iris_tree_entropy.png')
plt.tight_layout()
plt.show()