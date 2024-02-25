import pandas as pd


def augment_data(wiki_df):
    balanced_wiki_df = balance_genders(wiki_df)
    

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

    return pd.concat(balanced_samples)
