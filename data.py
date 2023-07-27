# data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from config import PARAMS


def load_data(data_path):
    df = pd.read_csv(data_path)

    # Preprocess userId and movieId columns
    df["userId"] = encode_user_ids(df["userId"])
    df["movieId"] = encode_movie_ids(df["movieId"])

    return df


def encode_user_ids(user_ids):
    encoder = LabelEncoder()
    return encoder.fit_transform(user_ids)


def encode_movie_ids(movie_ids):
    encoder = LabelEncoder()
    return encoder.fit_transform(movie_ids)


def get_dataloaders(df):
    train, val = train_test_split(df, test_size=PARAMS["test_size"])

    train_loader = DataLoader(train, batch_size=PARAMS["batch_szie"])
    val_loader = DataLoader(val, batch_size=PARAMS["batch_size"])

    return train_loader, val_loader
