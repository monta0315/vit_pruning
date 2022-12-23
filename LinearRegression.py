# from utils.utils import linear_regression => if use
import math

import numpy as np
from sklearn.linear_model import LinearRegression

from utils.utils import txt_impotance_scores_convert_array

importance_score_lists = txt_impotance_scores_convert_array("base-CIFAR10-100epochs-256bs-each-each",0)

x = np.array(importance_score_lists[2])
y = np.array(importance_score_lists[1])
print("corrcoef",np.corrcoef(x,y))
# 転置
x = np.array(importance_score_lists[2]).reshape((-1,1))
model = LinearRegression()
model.fit(x, y)
r_sq = model.score(x, y)
# 決定係数を出す意味があるのかはわからん => あんま関係ない気もする
print(f"coefficient of determination: {r_sq}")
y_h_arr = model.predict(x)
y_loss_arr = []
for i in range(len(x)):
    # np.arrayで下の数式を計算するのとただのarrayで計算するのだと計算結果が微妙に違う
    y_loss = math.fabs(y_h_arr[i]-y[i])
    y_loss_arr.append(y_loss)

print(y_loss_arr)