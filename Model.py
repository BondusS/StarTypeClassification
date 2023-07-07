import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

Dataset = pd.read_csv("Stars.csv")
Dataset[['Color', 'Spectral_Class']] = Dataset[['Color', 'Spectral_Class']].apply(lambda x: pd.factorize(x)[0])
Y_df = Dataset['Type']
X_df = Dataset.drop(['Type'], axis=1)
X_np = X_df.to_numpy()
Y_np = Y_df.to_numpy()
scaler = StandardScaler().fit(X_np)
X_st = scaler.transform(X_np)
x_train, x_test, y_train, y_test = train_test_split(X_st, Y_np)
Model = KNeighborsClassifier().fit(x_train, y_train)
print('Доля правильных ответов на обучающей выборке ', Model.score(x_train, y_train))
print('Доля правильных ответов на тестовой выборке ', Model.score(x_test, y_test))
print('Подборка лучших параметров...')
params = {'n_neighbors': [i for i in range(1, 51)],
          'weights': ['uniform', 'distance'],
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid = GridSearchCV(estimator=Model, param_grid=params)
grid.fit(x_train, y_train)
print('Наилучшая доля при подборке наиболее подходящих параметров ', grid.best_score_)
print('Наиболее подходящее количество соседей', grid.best_estimator_.n_neighbors)
print('Наилучшая весовая функция', grid.best_estimator_.weights)
print('Наилучший алгоритм поиска ближайших соседей', grid.best_estimator_.algorithm)
# Тестируем с наилучшими подобранными параметрами
BestModel = KNeighborsClassifier(n_neighbors=grid.best_estimator_.n_neighbors,
                                 weights=grid.best_estimator_.weights,
                                 algorithm=grid.best_estimator_.algorithm).fit(x_train, y_train)
print('Доля правильных ответов на обучающей выборке ', BestModel.score(x_train, y_train))
print('Доля правильных ответов на тестовой выборке ', BestModel.score(x_test, y_test))
