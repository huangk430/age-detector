import tensorflow as tf
import numpy as np
import shutil
import boto3
import os
import tempfile


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from keras.preprocessing import image
from keras.models import load_model



app = FastAPI()

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MODEL_KEY = os.environ.get('MODEL_KEY')


s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

#Load the model from s3
with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
    # Download the file from S3 to the temporary file
    s3.download_file(BUCKET_NAME, MODEL_KEY, temp_file.name)

    model = load_model(temp_file.name)


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

            img = image.load_img(temp_file.name, target_size=(180, 180))
            img_ary = image.img_to_array(img)
            img_ary = np.expand_dims(img_ary, axis=0)
            img_ary /= 255.0

            response = model.predict(img_ary)
            predicted_age = int(response[0][0])
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