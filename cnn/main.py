import pandas as pd

from load_dataset import load_dataset
from data_augmentation import augment_data, balance_genders
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle




def main():
    df = load_dataset()
    df = balance_genders(df)

    X = df
    y = df['Age']

    # Split data into training (80%) , validation (10%) and test sets (10%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    train_generator, val_generator = augment_data(train_df=X_train, val_df=X_val)

    


if __name__ == "__main__":
    main()