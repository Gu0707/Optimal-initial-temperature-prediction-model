import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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
logs = Logs("v3")


def run_randomforest(X_train, X_test, y_train, y_test, n_est, min_sp, dep, random_state=random.randint(0, 10000000)):
    # 创建随机森林分类器实例并训练
    model = RandomForestRegressor(n_estimators=n_est, min_samples_split=min_sp, max_depth=dep, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    train_predictions = model.predict(X_train)
    train_r2 = r2_score(y_train, train_predictions)

    # CLASS
    # model = RandomForestClassifier(n_estimators=200, min_samples_split=7, max_depth=5, random_state=888)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # score = accuracy_score(y_test, predictions)
    #
    # predictions = model.predict(X_train)
    # train_score = accuracy_score(y_train, predictions)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, # "Cla": score,
            "train_r2": train_r2, # "train_cla": train_score
            }


def run_mlp(X_train, X_test, y_train, y_test, hidden_neus, random_state=random.randint(0, 10000000)):
    # 创建随机森林分类器实例并训练
    model = MLPRegressor(hidden_layer_sizes=hidden_neus, max_iter=10000, random_state=random_state)
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

    correlations[sheet] = {}
    pearson_corr = df.corr(method='pearson')
    for i, col in enumerate(df.columns[:-1]):
        correlations[sheet][col] = pearson_corr.iloc[i, -1]
    logs.print(pearson_corr.iloc[:, -1])

    # 切分特征与目标
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    scaler = MinMaxScaler()

    # 切分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=91)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)

    # record
    results[sheet] = {}

    for n_est in tqdm([5, 10, 20, 50, 100, 200, 500]):
        for min_sp in tqdm([2, 3, 4, 5, 6, 7, 8, 9, 10], leave=False):
            for max_dep in tqdm([4, 5, 6, 7, 8, 9, 10], leave=False):
                for i in tqdm(range(20), leave=False):
                    random_state = random.randint(0, 10000000)
                    results[sheet][f"RandomForest-n{n_est}-sp{min_sp}-dep{max_dep}-r{random_state}"] = run_randomforest(X_train, X_test, y_train, y_test, n_est, min_sp, max_dep, random_state)
                    # logs.print(f'{sheet}-RandomForest-n{n_est}-sp{min_sp}-dep{max_dep}-r{random_state}: ', {key: f"{value:.4f}" for key, value in results[sheet][f"RandomForest-n{n_est}-sp{min_sp}-dep{max_dep}-r{random_state}"].items()})

    for hidden_neus in tqdm([(128, ), (256, ), (512, ), (1024, ),
                        (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024),
                        (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256),
                        (32, 64, 32), (64, 128, 64), (128, 256, 128), (256, 512, 256),
                        (32, 128, 32), (64, 256, 64), (128, 512, 128)]):
        for i in tqdm(range(20), leave=False):
            random_state = random.randint(0, 10000000)
            results[sheet][f"MLP-neu{hidden_neus}-r{random_state}"] = run_mlp(X_train, X_test, y_train, y_test, hidden_neus, random_state)
            # logs.print(f'{sheet}-MLP-neu{hidden_neus}-r{random_state}: ', {key: f"{value:.4f}" for key, value in results[sheet][f"MLP-neu{hidden_neus}-r{random_state}"].items()})

# 输出每个表的预测准确率
json.dump(results, open("results-v3.json", 'w'))
json.dump(correlations, open("correlations-v3.json", 'w'))
logs.print("指标:")
for sheet, accuracy in results.items():
    logs.print(f'\nTest accuracy for {sheet}:')
    for key_acc in accuracy:
        logs.print(f'{key_acc}: ', {key: f"{value:.4f}" for key, value in accuracy[key_acc].items()})