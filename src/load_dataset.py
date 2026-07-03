from datasets import load_dataset
from config import DATASET_NAME

def get_dataset():
    dataset = load_dataset(DATASET_NAME)
    return dataset

ex = get_dataset()
print (ex)