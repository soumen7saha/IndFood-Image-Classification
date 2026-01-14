import os
import shutil
from typing import Annotated
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from src.scripts.predict import *

app = FastAPI(title="IndFood Classification")
app.mount("/static", StaticFiles(directory='static'), name="static")
UPLOAD_DIR = "static/uploads"
UPLOAD_FILE: str = ''
os.makedirs(UPLOAD_DIR, exist_ok=True)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, imgFile: UploadFile = File(...)):
    # file parameter name is taken from the index.html file
    file_location = os.path.join(UPLOAD_DIR, imgFile.filename)
    with open(file_location, 'wb') as f:
        shutil.copyfileobj(imgFile.file, f)
    
    file_info = {
        'filename': imgFile.filename,
        'content_type': imgFile.content_type,
        'image_url': f"static/uploads/{imgFile.filename}" 
    }

    global UPLOAD_FILE
    UPLOAD_FILE = file_info.get('image_url', '')

    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'file_info': file_info,
        }
    )


@app.post("/classify_food", response_class=HTMLResponse)
def predict_food(
    request: Request, 
    img_url: str = None,
    model: str = Form(...)
):
    if img_url is None:
        img_url = UPLOAD_FILE
    if not os.path.exists(img_url):
        raise HTTPException(status_code=404, detail="File not found")

    fm_req_obj = FoodModel(img_url=img_url, model=model)

    return templates.TemplateResponse(
        'index.html',
        {
            "request": request,
            "result": predict(fm_req_obj),
            "file_info": {'image_url': img_url}
        }
    )


@app.post("/predict_food", response_model=ClassResponse)
def predict_endpoint(request: FoodModel):
    return predict(request)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
    # clear the upload_img folder when the app is stopped
    if len(os.listdir(UPLOAD_DIR)) != 0:
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
