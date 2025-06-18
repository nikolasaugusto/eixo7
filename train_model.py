import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv('data.csv', sep=';')

df = df.drop(columns=['date'])  
df = df.dropna(subset=['price'])  

def limpar_valores_numericos(valor):
    if isinstance(valor, str):
        if 'ou mais' in valor:
            return int(valor.split(' ')[0]) 
        elif valor.isdigit():
            return int(valor)
        else:
            return np.nan  
    return valor

colunas_com_texto_numerico = ['Quartos', 'Banheiros', 'Vagas na garagem']
for col in colunas_com_texto_numerico:
    if col in df.columns:
        df[col] = df[col].apply(limpar_valores_numericos)

colunas_categoricas = ['title', 'location', 'destaque', 'Categoria', 'Tipo', 'Detalhes do imóvel',
                       'Detalhes do condomínio', 'Zona', 'bairro']

colunas_numericas = ['oldPrice', 'Condomínio', 'Área útil', 'Quartos', 'Banheiros', 'Vagas na garagem',
                     'Academia', 'Elevador', 'Permitido animais', 'Piscina', 'Portaria', 'Salão de festas',
                     'Portão eletrônico', 'Área murada', 'Área de serviço', 'Armários na cozinha',
                     'Armários no quarto', 'Churrasqueira', 'Mobiliado', 'Quarto de serviço',
                     'Ar condicionado', 'Porteiro 24h', 'Varanda', 'IPTU']

colunas_existentes = [col for col in colunas_categoricas + colunas_numericas if col in df.columns]
X = df[colunas_existentes]
y = df['price']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, [col for col in colunas_numericas if col in df.columns]),
    ('cat', cat_pipeline, [col for col in colunas_categoricas if col in df.columns])
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X, y)

joblib.dump(model, 'modelo.joblib')

print("Modelo treinado e salvo como 'modelo.joblib'")
