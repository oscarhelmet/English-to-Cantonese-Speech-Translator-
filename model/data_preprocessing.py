from datasets import load_dataset
import pandas as pd


def data_preprocessing():
    # Source: https://huggingface.co/datasets/alt
    dataset = load_dataset('alt' , cache_dir=None)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)

    print(train_df.head())
    print(train_df.info())
    print(train_df.isnull().sum())

    train_df.fillna('Missing', inplace=True)
    test_df.fillna('Missing', inplace=True)


    train_df.to_csv('train_dataset.tsv', sep='\t', index=False)
    print("Successfully loaded training dataset in train_dataset.tsv")
    test_df.to_csv('test_dataset.tsv', sep='\t', index=False)
    print("Successfully loaded test dataset in test_dataset.tsv")

    return train_dataset, test_dataset