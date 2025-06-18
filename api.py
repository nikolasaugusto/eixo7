from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="API de Previsão de Imóveis", version="1.0")

try:
    model_rf = joblib.load('modelo.joblib')
    model_gradient = joblib.load('modelo_gradient.joblib')
    model_linear = joblib.load('modelo_linear.joblib') 
except Exception as e:
    raise RuntimeError(f"Erro ao carregar modelos: {e}")

class ImovelInput(BaseModel):
    oldPrice: float | None = None
    Condomínio: float | None = None
    Área_útil: float | None = None
    Quartos: float | None = None
    Banheiros: float | None = None
    Vagas_na_garagem: float | None = None
    Academia: float | None = None
    Elevador: float | None = None
    Permitido_animais: float | None = None
    Piscina: float | None = None
    Portaria: float | None = None
    Salão_de_festas: float | None = None
    Portão_eletrônico: float | None = None
    Área_murada: float | None = None
    Área_de_serviço: float | None = None
    Armários_na_cozinha: float | None = None
    Armários_no_quarto: float | None = None
    Churrasqueira: float | None = None
    Mobiliado: float | None = None
    Quarto_de_serviço: float | None = None
    Ar_condicionado: float | None = None
    Porteiro_24h: float | None = None
    Varanda: float | None = None
    IPTU: float | None = None
    title: str | None = None
    location: str | None = None
    destaque: str | None = None
    Categoria: str | None = None
    Tipo: str | None = None
    Detalhes_do_imóvel: str | None = None
    Detalhes_do_condomínio: str | None = None
    Zona: str | None = None
    bairro: str | None = None

def preprocess_input(input_data: ImovelInput) -> pd.DataFrame:
    df = pd.DataFrame([input_data.dict()])
    df.columns = [col.replace("_", " ") for col in df.columns]
    return df

def preprocess_input_linear(input_data: ImovelInput) -> pd.DataFrame:
    df = pd.DataFrame([input_data.dict()])
    df.columns = [col.replace("_", " ") for col in df.columns]

    expected_columns = model_linear.feature_names_in_

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    return df

@app.post("/predict", tags=["Random Forest"])
def predict_rf(input_data: ImovelInput):
    try:
        data = preprocess_input(input_data)
        prediction = model_rf.predict(data)[0]
        return {"modelo": "RandomForest", "preco_previsto": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

@app.post("/predict/gradient", tags=["Gradient Boosting"])
def predict_gradient(input_data: ImovelInput):
    try:
        data = preprocess_input(input_data)
        prediction = model_gradient.predict(data)[0]
        return {"modelo": "GradientBoosting", "preco_previsto": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

@app.post("/predict/linear", tags=["Regressão Linear"])
def predict_linear(input_data: ImovelInput):
    try:
        data = preprocess_input_linear(input_data)
        prediction = model_linear.predict(data)[0]
        return {"modelo": "LinearRegression", "preco_previsto": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {e}")

@app.get("/")
def home():
    return {"mensagem": "API de Previsão de Imóveis está online. Acesse /docs para usar."}
