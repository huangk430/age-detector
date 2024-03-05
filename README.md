# Age Detector

### Dataset
Create a new folder called dataset in the server 
```bash
cd server
mkdir dataset
cd dataset
```
Download [IMDB Faces File](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar) and[WIKI Faces File](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar),

Drop the .tar files into the "dataset" folder.

Untar the files.
```bash
tar -xvf imdb_crop.tar
tar -xvf wiki_crop.tar
```

### Model
Our model is too large to save into GitHub, so we save it into Google Drive. To use our model, download the files from this [link](https://drive.google.com/drive/folders/16PHv37i6Ahb9lA-1qu0fN4tUkBX_QT9R?usp=sharing), and drop it into the cnn/model-versions/v1 folder.