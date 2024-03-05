import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


def augment_data(train_df, val_df):

    train_datagen=ImageDataGenerator(
        rescale=1./255.,
        rotation_range=25,
        brightness_range=[0.7,1.3],
        zoom_range=0.5,
        horizontal_flip=True
    )    

    valid_datagen = ImageDataGenerator(rescale=1./255.) 

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='server/dataset/wiki_crop',
        x_col="Image Path",
        y_col="Age",
        batch_size=64,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(180,180)
    )

    val_generator=valid_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory="server/dataset/wiki_crop",
        x_col="Image Path",
        y_col="Age",
        batch_size=64,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(180,180)
    )
    
    return (train_generator, val_generator)


def balance_genders(df) -> pd.DataFrame:

    male_data = df[df["Gender"] == 1]
    female_data = df[df["Gender"] == 0]

    # Dividing the data into age groups
    bins = [0, 20, 30, 40, 50, 100]
    labels = ["0-20", "21-30", "31-40", "41-50", "51-100"]  # Making the bins reflect the IQR
    
    male_data.loc[:, 'age_group'] = pd.cut(male_data['Age'], bins=bins, labels=labels)
    female_data.loc[:, 'age_group'] = pd.cut(female_data['Age'], bins=bins, labels=labels)

    balanced_samples = []

    # Performing stratified sampling within each age group
    for age_group in labels:
        male_subset = male_data[male_data['age_group'] == age_group]
        female_subset = female_data[female_data['age_group'] == age_group]
        
        # Determine the number of samples needed to balance the classes
        num_samples = min(len(male_subset), len(female_subset))
        
        # Sample the same number of instances from both male and female subsets
        male_sample = male_subset.sample(n=num_samples, random_state=42)
        female_sample = female_subset.sample(n=num_samples, random_state=42)
        
        # Combine the balanced samples
        balanced_samples.extend([male_sample, female_sample])

    df = pd.concat(balanced_samples)
    df = df.drop(columns=['age_group'])

    return df
