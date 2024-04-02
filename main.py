# main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from semantic_similarity import predict

app = FastAPI()

# Mount the 'static' directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static")

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/predict_similarity/")
async def predict_similarity(request: Request, text_pair: TextPair):
    try:
        similarity_score = predict(text_pair.text1, text_pair.text2)
        return JSONResponse({"similarity_score": similarity_score})
        # return templates.TemplateResponse("result.html", {"request": request, "similarity_score": similarity_score})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.get("/items/{id}", response_class=HTMLResponse)
# async def read_item(request: Request, id: str):
#     return templates.TemplateResponse(
#         request=request, name="item.html", context={"id": id}
#     )
