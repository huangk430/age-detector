from load_dataset import load_dataset
from data_augmentation import augment_data


def main():
    wiki_df = load_dataset()
    augmented_wiki_df = augment_data(wiki_df)
    



if __name__ == "__main__":
    main()