"""サイキットラーンの標準データの手書き数字の8×8画像データを、トレーニングデータとテストデータに分割した。そして、モデルを学習したのち、計算した正解率と予測結果（一部）を表示するようにした。
ロジスティック回帰、SVM、ランダムフォレストで行い比較した。
SVMが一番精度が高かった。
サンプルも出力し、真のラベル、各分類器の結果を表示するようにした。"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# ロジスティック回帰のトレーニング
logreg = LogisticRegression(max_iter=5000, solver="lbfgs")
logreg.fit(X_train, y_train)
logreg_y_pred = logreg.predict(X_test)
logreg_acc = accuracy_score(y_test, logreg_y_pred)

# サポートベクターマシンのトレーニング
svc = SVC()
svc.fit(X_train, y_train)
svc_y_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_y_pred)

# ランダムフォレストのトレーニング
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_y_pred)

# 結果
print("Accuracy - Logistic Regression: ", logreg_acc)
print("Accuracy - SVM: ", svc_acc)
print("Accuracy - Random Forest: ", rf_acc)

# サンプル出力
plt.figure(figsize=(20,6))
for i in range(30):
    plt.subplot(3, 10, i+1)
    plt.imshow(X_test[i].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    plt.title("True: %d\nLR: %d\nSVC: %d\nRF: %d" % (y_test[i], logreg_y_pred[i], svc_y_pred[i], rf_y_pred[i]))

plt.show()
