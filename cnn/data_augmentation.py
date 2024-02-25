from PIL import Image

from cnn.load_dataset import load_dataset


def augment_data(wiki_df, imdb_df):
    # wiki_df['cropped_image'] = wiki_df.apply(lambda row: crop_face(row['image_path'], row['face_location'][0], row['y'], row['width'], row['height']), axis=1)
    # imdb_df['cropped_image'] = imdb_df.apply(lambda row: crop_face(row['image_path'], row['x'], row['y'], row['width'], row['height']), axis=1)
    img = Image.open('01/nm0000001_rm3343756032_1899-5-10_1970.jpg')
    face_region = img.crop((477.184, 100.352, 622.592, 245.76))

    face_region.show()


