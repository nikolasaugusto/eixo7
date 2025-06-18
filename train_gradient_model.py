import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import joblib

df = pd.read_csv('data.csv', sep=';')

df = df.drop(columns=['date'], errors='ignore')  # 'date' tem dados inválidos
df = df.dropna(subset=['price'])  # 'price' é a variável alvo

for col in ['Quartos', 'Banheiros', 'Vagas na garagem']:
    if col in df.columns:
        df[col] = df[col].replace('5 ou mais', 5).astype(float)

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
    ('regressor', GradientBoostingRegressor(random_state=42))
])

model.fit(X, y)

joblib.dump(model, 'modelo_gradient.joblib')

print("Modelo Gradient Boosting treinado e salvo como 'modelo_gradient.joblib'")
