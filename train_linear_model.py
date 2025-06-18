import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('data.csv', sep=';')

cols_to_fix = ['Quartos', 'Banheiros', 'Vagas na garagem']
for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col].replace('5 ou mais', 5).astype(float)

df.fillna(0, inplace=True)

features = ['oldPrice', 'Condomínio', 'Área útil', 'Quartos', 'Banheiros', 'Vagas na garagem',
            'Academia', 'Elevador', 'Permitido animais', 'Piscina', 'Portaria', 'Salão de festas',
            'Portão eletrônico', 'Área murada', 'Área de serviço', 'Armários na cozinha', 'Armários no quarto',
            'Churrasqueira', 'Mobiliado', 'Quarto de serviço', 'Ar condicionado', 'Porteiro 24h',
            'Varanda', 'IPTU']

features = [f for f in features if f in df.columns]

X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

y_pred = model_linear.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

joblib.dump(model_linear, 'modelo_linear.joblib')
print("Modelo linear salvo em 'modelo_linear.joblib'")
