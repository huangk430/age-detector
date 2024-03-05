import tensorflow as tf
import numpy as np
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from keras.preprocessing import image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"]
)

@app.get("/")
async def health_check():
    return {"status": "ok"}


@app.post("/predict-age")
async def predict_age(file: UploadFile = File(...)):

    try:
        with NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)

            model = tf.saved_model.load("server/cnn/model_versions/v1")
            img = image.load_img(temp_file.name, target_size=(180, 180))
            img_ary = image.img_to_array(img)
            img_ary = np.expand_dims(img_ary, axis=0)
            img_ary /= 255.0

            predicted_age = int(model(img_ary).numpy()[0][0])
            return {"predicted_age": predicted_age}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error predicting age: {str(e)}")
    
    finally:
        if temp_file:
            temp_file.close()
            shutil.rmtree(temp_file.name, ignore_errors=True)
    

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)