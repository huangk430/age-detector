import tensorflow as tf

from fastapi import FastAPI, File, UploadFile


app = FastAPI()


@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/predict-age")
async def predict_age(file: UploadFile = File(...)):
    # contents = await image.read()
    v1_model_path = "cnn/model-versions/v1/saved_model.pb"
    v1_model = tf.saved_model.load(v1_model_path)
    output = model(file)

    
    return {"filename": image.filename}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)