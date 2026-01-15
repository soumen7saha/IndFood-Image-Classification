# import the packages
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn
from src.scripts.model import *


class FoodModel(BaseModel):
    img_url: Path = Field(..., json_schema_extra={"examples": ["src/images/masala_dosa.jpg"]})
    model: str = Field(..., json_schema_extra={"examples": ["convns", "resnet"]})


class ClassResponse(BaseModel):
    t1_class: str
    t5_preds: dict


app = FastAPI(title="indfood-prediction")

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/classify_food")
def predict(req: FoodModel) -> ClassResponse:
    result = None
    if req.model == 'resnet':
        result = resnet(req.img_url)
    elif req.model == 'convns':
        result = convns(req.img_url)
    
    return ClassResponse(
        t1_class = result['t1_class'],
        t5_preds = result['t5_preds']
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)