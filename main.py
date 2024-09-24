import pandas as pd
import matplotlib.pyplot as plt
#from category_encoders.one_hot import OneHotEncoder, OrdinalEncoder
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from pycaret.regression import *

# ------------- Modelo de Classificação

"""iris = load_iris()
#print(iris)
#print(iris.target[[10, 25, 50]])
#print(list(iris.target_names))
#print(iris.data) # features dos dados
#print(iris.target) # classificação dos dados
#iris.data

X,y = load_iris(return_X_y=True) # X é o data e y é o target, semelhante a y = f(X)
#print(X) # iris.data
#print(y) # iris.target
iris_df = pd.DataFrame(data=X, columns= iris.feature_names)
iris_df['target'] = y
iris_df['target names'] = pd.Categorical.from_codes(y,iris.target_names)
#print(iris_df)
iris_df.plot.scatter('sepal length (cm)', 'sepal width (cm)', c='target')

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.40,random_state=13)
print(X_train.shape)
print(X_test.shape)
classificador = svm.SVC(kernel='linear',C=1).fit(X_train,y_train)
print(classificador.score(X_test,y_test))
y_pred = classificador.predict(X_test)
print(classification_report(y_test,y_pred,target_names=iris.target_names))"""

# ------------ Modelo de Regressão

"""diab = load_diabetes()
print(diab)
print(diab.target[:6]) # as 6 primeiras classificações
print(diab.data.shape)
X,y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.60,random_state=13)
reg = LinearRegression().fit(X_train,y_train)
y_pred = reg.predict(X_test)
#print(y_pred)
print(reg.score(X_test,y_test))
print(reg.coef_)
print(reg.intercept_)
print(X_test.shape)
print(y_pred.shape)
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()"""


# -------------- Variáveis categóricas

"""df_enem = pd.read_csv('MICRODADOS_ENEM_2022.csv',encoding= "ISO-8859-1",sep=';')
#print(df_enem.head())
#print(df_enem.info(verbose=True)) #informações sobre a tabela
#print(df_enem.select_dtypes(include='object').describe()) # pegando a distribuição categorica dessa cas colunas com tipos de dados object
one_hot_encoder = OneHotEncoder(cols=['TP_SEXO']) # TRANSFORMANDO A COLUNA SEXO QUE POSSUI 2 RÓTULOS, MASCULINO E FEMININO, EM 2 COLUNAS. Quem for do sexo masculino, por exemplo, terá 1 na linha para indicar q é do sexo masculino naquela coluna do sexo masculino
df_enem_categoricas = df_enem.select_dtypes(include='object') # recebe
df_enem_categoricas_oneHot =  one_hot_encoder.fit_transform(df_enem_categoricas)
#print(df_enem_categoricas_oneHot.columns) # mostrando as colunas que tem nessa tabela
ord_enc = OrdinalEncoder(cols=['Q001', 'Q002', 'Q003', 'Q004', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010',
       'Q011', 'Q012', 'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019',
       'Q020', 'Q021', 'Q022', 'Q023', 'Q024', 'Q025']) # vai trocar letras, ou categorias, que tem ordenação por números. Por exemplo, Q025 vai ter 1 onde tem a letra A, 2 onde tem a letra B.
print(ord_enc.fit_transform(df_enem_categoricas_oneHot))"""

# -------------- Treinando com exemplos

"""df_ipca = pd.read_excel('ipca_202405SerieHist.xls')
#print(df_ipca.head(10))
#print(df_ipca.info(verbose=True))
#print(df_ipca.columns)
#print(df_ipca.describe())
#print(df_ipca.isna().sum()) # mostra quantos valores faltantes tem em cada coluna
#df_ipca.dropna() # remove as linhas com valores faltantes
#df_ipca.drop_duplicates() # remove duplicatas
#print(df_ipca.loc[df_ipca['Unamed; 1'] != 0]) # mostrando as linhas que possuem valor diferente de 0 na coluna unamed
df_ipca2 = df_ipca.drop(columns=['Unnamed: 0', 'Unnamed: 1']) # retirando a coluna Unnamed: 0
df_ipca2 = df_ipca2.dropna()
df_ipca2 = df_ipca2.drop([4,77,150,223,296,369], axis='index')  # removendo a linha 4 do dataframe 4, 77,
#df_ipca2.to_csv('out.csv', index=False) # transformando um dataframe em csv
#print(df_ipca2['Unnamed: 2'].head(305))
#print(df_ipca2.tail(20)) # vendo os ultimos n registros
#df_ipca3 = df_ipca2[0]
#df_ipca3 = df_ipca3.drop([77,150,223,296,369], axis='index')
#df_ipca3.to_csv('out.csv', index=False)
#print(df_ipca3.head())
X = df_ipca2[['Unnamed: 2', 'Unnamed: 3','Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']]
y = df_ipca2['Unnamed: 7']
reg = LinearRegression().fit(X, y)
#print(reg.score(X, y))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.60,random_state=13)
reg = LinearRegression().fit(X_train,y_train)
y_pred = reg.predict(X_test)
#print(y_pred)
print(reg.score(X_test,y_test)) # Verifica se o modelo funciona satisfatoriamente
print(reg.coef_) # Coeficientes usados na previsão de cada alvo, ou seja, nas instâncias de x para achar o y correspondente joga-se numa fórmula e os coeficientes ligados ao x para achar o y são retornados nessa chamada
print(reg.intercept_)  # Quando x for 0 essa função retornará o valor estimado do y
#print(X_test.shape)
#print(y_pred.shape)
plt.scatter(X_test['Unnamed: 2'], y_test, color='black')  # Gráfico de dispersão
plt.plot(X_test['Unnamed: 2'], y_pred, color='black', linewidth=1)  # Desenha pontos marcadores em um diagrama
plt.scatter(X_test['Unnamed: 3'], y_test, color='orange')  # Gráfico de dispersão
plt.plot(X_test['Unnamed: 3'], y_pred, color='orange', linewidth=1)
plt.scatter(X_test['Unnamed: 4'], y_test, color='green')  # Gráfico de dispersão
plt.plot(X_test['Unnamed: 4'], y_pred, color='green', linewidth=1)
plt.scatter(X_test['Unnamed: 5'], y_test, color='red')  # Gráfico de dispersão
plt.plot(X_test['Unnamed: 5'], y_pred, color='red', linewidth=1)
plt.scatter(X_test['Unnamed: 6'], y_test, color='purple')  # Gráfico de dispersão
plt.plot(X_test['Unnamed: 6'], y_pred, color='purple', linewidth=1)
plt.show()"""


#  ----------------------  Pipelines - Processos com etapas que precisam ser aplicados em cada coluna - e Validação Cruzada

#X_treino = pd.read_csv('train.csv', index_col='Id')
#X_teste = pd.read_csv('test.csv', index_col='Id')
#print(X_treino.info(verbose=True))
"""X = X_treino.drop(['SalePrice'], axis=1)
feature_names = X_treino.columns.values
y = X_treino.SalePrice
ordinal_features = ['ExterCond', 'BsmtQual','BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu','GarageQual', 'GarageCond','PoolQC']
nominal_features = list(set(X.select_dtypes(include='object').columns.values) - set(ordinal_features))
numerical_features = X.select_dtypes(exclude='object').columns.values
numerical_transformer = SimpleImputer(strategy='median')    # Estratégias geralmente escolhidas para preenchimento de dados nulos são a mediana ou a moda, nessa cao foi escolhido a mediana
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat_nom', nominal_transformer, nominal_features),
        ('ord_nom', ordinal_transformer, ordinal_features)
    ])
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('minmaxscaler', MinMaxScaler()),
    ('stdscaler', StandardScaler()),
    ('regressor', DecisionTreeRegressor(random_state=0, criterion='friedman_mse'))
])  # Passos, respectivamente: substituição dos valores ausentes; normalização dos dados, ou seja, colocar os dados entre 0 e 1; padronização dos dados, processo de colocar os dados com média 0 e variancia 1, ou seja, ele dimimui a média e dividi pela variancia; instanciação do modelo, nesse caso regressao, random_satate é a semente para que ao rodar multiplas vezes tenhamos o mesmo modelo.
X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=0.8,random_state=3)
pipe.fit(X_train, y_train)
val_predictions = pipe.predict(X_val)
print(mean_squared_error(y_val, val_predictions, squared=False))  # Erro quadrático
print(mean_absolute_error(y_val, val_predictions))  # Erro absoluto
print(pd.DataFrame({'y_val': y_val,
              'val_pred': val_predictions,
              'abs_err': abs(val_predictions - y_val)}).describe())
scores = -1 * cross_val_score(pipe, X, y, cv=KFold(n_splits=5), scoring='neg_mean_absolute_error')  # Validação Cruzada, é um outro método de validação do modelo que ao invés de separar um dado de teste e um dado de treino, ele separa os dados em partes e a cada rodada pega uma parte e usa como teste, nessa caso sao 5 parte e 5 rodadas
print(scores)
print(scores.mean())  # Média dos scores, isso da mais ou menos o .score() da validação holdout, que é a validação que fizemos anteriormente a essa
print(scores.std()) # Desvio padrão dos scores


#  -----------------------  Ajueste de Hiperparametros - Parametros que não são passados por padrão, ou seja, passamos eles no modelo para que, de acordo com os ajustes, possamos ter um melhor resultado

espaço_de_parametros = dict(
 regressor__max_depth= [2,3,5],
 regressor__min_samples_split= [32,64,128],
 regressor__min_samples_leaf= [32,64,128]
)  # Alguns dos hiperparametros
busca_exaustiva = GridSearchCV(pipe, espaço_de_parametros, cv=KFold(n_splits=5))  #  Dentro daqueles parametros passados, no espaço de parametros, e no modelo pipe(poderiamos colocar ali o DecisionTreeRegressor, por exemplo, caso nao tenhamos pipe), escolhera a melhor combinação de parametros
busca_exaustiva.fit(X,y)
resultado = pd.DataFrame(busca_exaustiva.cv_results_)
print(resultado.head())
print(busca_exaustiva.best_estimator_)
print(busca_exaustiva.best_params_)
scores = -1 * cross_val_score(busca_exaustiva, X, y, cv=KFold(n_splits=5), scoring='neg_mean_absolute_error')  # Vai retornar os scores de acordo com a melhor combinação da busca exaustiva
print(scores)
print(scores.mean())
print(scores.std())
busca_randomizada = RandomizedSearchCV(pipe, espaço_de_parametros, cv=KFold(n_splits=5), n_iter=20, random_state=10)  # Dentro dos valores para os hiperparametros, ele vai no maximo fazer 20 combinações, porque o n_iter = 20, aleatorias de valores dos hiperparametro
busca_randomizada.fit(X,y)
resultado = pd.DataFrame(busca_randomizada.cv_results_)
print(resultado.head())
#print(busca_randomizada.best_estimator_)
print(busca_randomizada.best_params_)
scores = -1 * cross_val_score(busca_randomizada, X, y, cv=KFold(n_splits=5), scoring='neg_mean_absolute_error')
print(scores)
print(scores.mean())
print(scores.std()"""
X = pd.read_excel('32.xls', engine='xlrd')
print(X.columns)
#  ------------------------ AutoML com PyCaret -  Automatizando o processo de ML, ou seja,

"""regressor = setup(X_treino, remove_outliers=True, normalize=True, fold=5, target='SalePrice', silent=True)
#regressor = setup(X_treino, remove_outliers=True, normalize=True, fold=5, target='SalePrice', feature_selection=True, silent=True)  # Com o feature_selection, ele escolhe as melhores variaveis preditoras para o seu modelo
#print(regressor)
#melhor = compare_models(sort='RMSE')  # Ordenar pelo erro quadrático médio nessa comparação de modelos
#print(melhor)
huber_regressao = create_model('huber', fold=5)
print(huber_regressao)
tuned_regressao = tune_model(huber_regressao, fold=5)  # Tunning é o ajuste dos hiperparametros, parametros que o modelo não aprende quando ele está treinando sendo assim precisam ser inseridos para eles
print(tuned_regressao)
predicao = predict_model(tuned_regressao, data=X_teste)
print(predicao)"""