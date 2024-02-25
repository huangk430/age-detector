import os
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint
from IPython.display import display

def load_dataset() -> tuple(pd.DataFrame, pd.DataFrame):

    imdb_mat_path = "../dataset/imdb_crop/imdb.mat"
    wiki_mat_path = "../dataset/wiki_crop/wiki.mat"

    imdb_mat = sio.loadmat(imdb_mat_path)
    wiki_mat = sio.loadmat(wiki_mat_path)

    num_imdb_images = len(imdb_mat["imdb"][0][0][0][0])
    num_wiki_image = len(wiki_mat["wiki"][0][0][0][0])
    imdb_mat_content = imdb_mat["imdb"][0][0]
    wiki_mat_content = wiki_mat["wiki"][0][0]

    imdb_dob = imdb_mat_content[0][0]  # date of birth (Matlab serial date number)
    imdb_photo_taken = imdb_mat_content[1][0]  # year when the photo was taken
    imdb_full_path = imdb_mat_content[2][0]  # path to file
    imdb_gender = imdb_mat_content[3][0]  # 0 for female and 1 for male, NaN if unknown
    imdb_name = imdb_mat_content[4][0]  # name of the celebrity
    imdb_face_location = imdb_mat_content[5][0]  # location of the face
    imdb_face_score = imdb_mat_content[6][0]  # detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
    imdb_second_face_score = imdb_mat_content[7][0]  # detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.

    wiki_dob = wiki_mat_content[0][0]  # date of birth (Matlab serial date number)
    wiki_photo_taken = wiki_mat_content[1][0]  # year when the photo was taken
    wiki_full_path = wiki_mat_content[2][0]  # path to file
    wiki_gender = wiki_mat_content[3][0]  # 0 for female and 1 for male, NaN if unknown
    wiki_name = wiki_mat_content[4][0]  # name of the celebrity
    wiki_face_location = wiki_mat_content[5][0]  # location of the face
    wiki_face_score = wiki_mat_content[6][0]  # detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
    wiki_second_face_score = wiki_mat_content[7][0]  # detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.

    imdb_df = pd.DataFrame({
        'Year of Birth': imdb_dob, 
        'Photo Year': imdb_photo_taken, 
        'Image Path': imdb_full_path, 
        'Gender': imdb_gender, 
        'Name': imdb_name,
        'Face Score': imdb_face_score, 
        'Second Face Score': imdb_second_face_score
        })

    wiki_df = pd.DataFrame({
        'Year of Birth': wiki_dob, 
        'Photo Year': wiki_photo_taken, 
        'Image Path': wiki_full_path, 
        'Gender': wiki_gender, 
        'Name': wiki_name,
        'Face Score': wiki_face_score, 
        'Second Face Score': wiki_second_face_score
        })

    #  converting numpy lists to the first entry
    imdb_df['Name'] = imdb_df['Name'].apply(lambda x: x[0])
    imdb_df['Image Path'] = imdb_df['Image Path'].apply(lambda x: x[0])

    #  dropping rows where image path doesn't exist
    wiki_df = wiki_df[wiki_df["Name"].apply(len) > 0]

    #  converting numpy lists to the first entry
    wiki_df['Name'] = wiki_df['Name'].apply(lambda x: x[0])
    wiki_df['Image Path'] = wiki_df['Image Path'].apply(lambda x: x[0])

    imdb_df["Year of Birth"] = imdb_df["Year of Birth"] // 365
    imdb_df["Age"] = imdb_df["Photo Year"] - imdb_df["Year of Birth"]
    display(imdb_df)

    wiki_df["Year of Birth"] = wiki_df["Year of Birth"] // 365
    wiki_df["Age"] = wiki_df["Photo Year"] - wiki_df["Year of Birth"]
    display(wiki_df)
    
    return clean_data(wiki_df, imdb_df)

def clean_data(wiki_df, imdb_df):

    #  Dropping duplicate entries
    imdb_df = imdb_df.drop_duplicates()
    wiki_df = wiki_df.drop_duplicates()

    #  Removing genders with the value NaN, meaning that gender is unknown
    imdb_df = imdb_df[(imdb_df["Gender"] == 1) | 
            (imdb_df["Gender"] == 0)]
    wiki_df = wiki_df[(wiki_df["Gender"] == 1) | 
            (wiki_df["Gender"] == 0)]

    #  Removing face scores with the value -inf, meaning that there was no face detected
    imdb_df = imdb_df[imdb_df['Face Score'] != -np.inf]
    wiki_df = wiki_df[wiki_df['Face Score'] != -np.inf]

    #  Removing entries with invalid ages
    imdb_df = imdb_df[(imdb_df["Age"] > 0) &
            (imdb_df["Age"] <= 100)]
    wiki_df = wiki_df[(wiki_df["Age"] > 0) &
            (wiki_df["Age"] <= 100)]

    #  Dropping entries with missing values
    imdb_df.dropna()
    wiki_df.dropna()

    #  Removing entries where birthday is unknown (birth year == 0)
    imdb_df = imdb_df[imdb_df["Year of Birth"] != 0]
    wiki_df = wiki_df[wiki_df["Year of Birth"] != 0]

    #  Removing entries with second face score, meaning multiple people in one image
    imdb_df = imdb_df[pd.isna(imdb_df['Second Face Score'])]
    display(imdb_df)

    wiki_df = wiki_df[pd.isna(wiki_df['Second Face Score'])]
    display(wiki_df)
    
    return (wiki_df, imdb_df)