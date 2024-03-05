import numpy as np
import scipy.io as sio
import pandas as pd

from IPython.display import display


def load_dataset() -> pd.DataFrame:
    
    wiki_mat_path = "server/dataset/wiki_crop/wiki.mat"

    wiki_mat = sio.loadmat(wiki_mat_path)

    wiki_mat_content = wiki_mat["wiki"][0][0]

    wiki_dob = wiki_mat_content[0][0]  # date of birth (Matlab serial date number)
    wiki_photo_taken = wiki_mat_content[1][0]  # year when the photo was taken
    wiki_full_path = wiki_mat_content[2][0]  # path to file
    wiki_gender = wiki_mat_content[3][0]  # 0 for female and 1 for male, NaN if unknown
    wiki_name = wiki_mat_content[4][0]  # name of the celebrity
    wiki_face_score = wiki_mat_content[6][0]  # detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
    wiki_second_face_score = wiki_mat_content[7][0]  # detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.

    wiki_df = pd.DataFrame({
        'Year of Birth': wiki_dob, 
        'Photo Year': wiki_photo_taken, 
        'Image Path': wiki_full_path, 
        'Gender': wiki_gender, 
        'Name': wiki_name,
        'Face Score': wiki_face_score, 
        'Second Face Score': wiki_second_face_score
        })

    #  dropping rows where image path doesn't exist
    wiki_df = wiki_df[wiki_df["Name"].apply(len) > 0]

    #  converting numpy lists to the first entry
    wiki_df['Name'] = wiki_df['Name'].apply(lambda x: x[0])
    wiki_df['Image Path'] = wiki_df['Image Path'].apply(lambda x: x[0])

    wiki_df["Year of Birth"] = wiki_df["Year of Birth"] // 365
    wiki_df["Age"] = wiki_df["Photo Year"] - wiki_df["Year of Birth"]
    
    return clean_data(wiki_df)

def clean_data(wiki_df):

    #  Dropping duplicate entries
    wiki_df = wiki_df.drop_duplicates()

    #  Removing genders with the value NaN, meaning that gender is unknown
    wiki_df = wiki_df[(wiki_df["Gender"] == 1) | 
            (wiki_df["Gender"] == 0)]

    #  Removing face scores with the value -inf, meaning that there was no face detected
    wiki_df = wiki_df[wiki_df['Face Score'] != -np.inf]

    #  Removing entries with invalid ages
    wiki_df = wiki_df[(wiki_df["Age"] > 0) &
            (wiki_df["Age"] <= 100)]

    #  Dropping entries with missing values
    wiki_df.dropna()

    #  Removing entries where birthday is unknown (birth year == 0)
    wiki_df = wiki_df[wiki_df["Year of Birth"] != 0]

    #  Removing entries with second face score, meaning multiple people in one image
    wiki_df = wiki_df[pd.isna(wiki_df['Second Face Score'])]
    
    return wiki_df
