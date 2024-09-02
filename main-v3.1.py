import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import random, datetime
from tqdm import tqdm

class Logs:
    def __init__(self, version=''):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.file = open(f"log-{version}-{current_datetime}.log", 'w')

    def print(self, *args, sep=' ', end='\n'):
        print(*args, sep=sep, end=end, flush=True)
        print(*args, sep=sep, end=end, flush=True, file=self.file)

    def __del__(self):
        self.file.close()


# v2读取4.4的数据

# 读取Excel文件，这里假设文件名为"data.xlsx"
xlsx = pd.ExcelFile('data.xlsx')
sheet_names = xlsx.sheet_names  # 获取所有表名
logs = Logs("v3-1")


def run_bagging(X_train, X_test, y_train, y_test, n_est, random_state=random.randint(0, 10000000)):
    # 创建随机森林分类器实例并训练
    model = BaggingRegressor(n_estimators=n_est, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    train_predictions = model.predict(X_train)
    train_r2 = r2_score(y_train, train_predictions)

    # CLASS
    # model = MLPClassifier(hidden_layer_sizes=(64, 64, 64), max_iter=5000, random_state=27777)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # score = accuracy_score(y_test, predictions)

    # predictions = model.predict(X_train)
    # train_score = accuracy_score(y_train, predictions)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, # "Cla": score,
            "train_r2": train_r2, # "train_cla": train_score
            }



# 初始化预测结果存储字典
results = {}
correlations = {}

for sheet in sheet_names:
    # 按表名读取数据
    df = xlsx.parse(sheet)

    # 去除非完整数据
    df = df.dropna()

    # correlations[sheet] = {}
    # pearson_corr = df.corr(method='pearson')
    # for i, col in enumerate(df.columns[:-1]):
    #     correlations[sheet][col] = pearson_corr.iloc[i, -1]
    # logs.print(pearson_corr.iloc[:, -1])

    # 切分特征与目标
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    scaler = MinMaxScaler()

    # 切分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=91)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    # record
    results[sheet] = {}
    
    for n_est in tqdm([20, 50, 100, 200, 300, 500]):
        for i in tqdm(range(20), leave=False):
            random_state = random.randint(0, 10000000)
            results[sheet][f"Bagging-nest{n_est}-r{random_state}"] = run_bagging(X_train, X_test, y_train, y_test, n_est, random_state)
            # logs.print(f'{sheet}-Bagging-nest{n_est}-r{random_state}: ', {key: f"{value:.4f}" for key, value in results[sheet][f"Bagging-nest{n_est}-r{random_state}"].items()})

# 输出每个表的预测准确率
json.dump(results, open("results-v3-1.json", 'w'))
logs.print("指标:")
for sheet, accuracy in results.items():
    logs.print(f'\nTest accuracy for {sheet}:')
    for key_acc in accuracy:
        logs.print(f'{key_acc}: ', {key: f"{value:.4f}" for key, value in accuracy[key_acc].items()})