from load_dataset import load_dataset
from data_augmentation import augment_data


def main():
    wiki_df, imdb_df = load_dataset()
    augment_data(wiki_df, imdb_df)
    



if __name__ == "__main__":
    main()