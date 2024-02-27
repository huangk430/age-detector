from fastapi import FastAPI, File, UploadFile


app = FastAPI()


@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/predict-age")
async def predict_age(image: UploadFile = File(...)):
    contents = await image.read()       
    
    return {"filename": image.filename}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)