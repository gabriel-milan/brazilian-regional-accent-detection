import os
import uuid
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from minio import Minio
import tensorflow as tf
from sklearn import preprocessing
from collections import namedtuple
from tensorflow.keras.models import Model

EXTRACTED_FEATURES_DATAFRAME = "data/extracted_features.csv"

Config = namedtuple("Config", [
    "data_path",
    "random_seed",
    "n_folds",
    "max_epochs",
    "optimizer",
    "loss_func",
    "monitor",
    "monitor_mode",
    "verbose",
    "lr_factor",
    "lr_patience",
    "es_patience",
    "batch_size",
    "feature_sets",
    "accents"
])


def load_environments(env_filename: str = ".env"):
    """
    Loads the environments from the .env file.
    :param env_filename: the .env file
    :return: None
    """
    with open(env_filename, "r") as f:
        env = {}
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            os.environ.setdefault(line.split("=")[0], line.split("=")[1])


def load_samples_from_file(filename: str, sample_duration: float = 0.02):
    """
    Load audio samples from a file.

    :param filename: path to the file
    :param sample_duration: duration of each sample in seconds
    :return: tuple of (audio samples, sample rate)
    """
    y, sr = librosa.load(filename, sr=44100)
    window_size = int(sample_duration * sr)
    samples = []
    for i in range(0, len(y), window_size):
        if (y[i:i + window_size].shape[0] == window_size):
            samples.append(y[i:i + window_size])
    samples = np.array(samples)
    return samples, sr


def generate_dataset_from_file_list(file_list: list, sample_duration: float = 0.02):
    """
    Generate a dataset from a list of audio files.

    :param file_list: list of paths to the files
    :param sample_duration: duration of each sample in seconds
    :return: tuple of (audio samples, sample rate)
    """
    samples, sr = load_samples_from_file(file_list[0], sample_duration)
    sample_rate = [sr]
    for file in tqdm(file_list[1:]):
        sample, sr = load_samples_from_file(file, sample_duration)
        samples = np.concatenate((samples, sample), axis=0)
        sample_rate.append(sr)
    samples = np.array(samples)
    return samples, sample_rate


def generate_dataset_from_df(df: pd.DataFrame, sample_duration: float = 0.02):
    """
    Generate a dataset from a dataframe.

    :param df: dataframe containing the paths to the files
    :param sample_duration: duration of each sample in seconds
    :return: tuple of (audio samples, sample rate)
    """
    file_list = ("data/" + df['path_wav']).tolist()
    samples, sample_rate = generate_dataset_from_file_list(
        file_list, sample_duration)
    return samples, sample_rate


def get_Xy_from_df(df: pd.DataFrame, sotaques: list, sample_duration: float = 0.02):
    """
    Get X and y from a dataframe.

    :param df: dataframe
    :param sotaques: list of sotaques
    :return: tuple of (X, y)
    """
    X = []
    y = []
    for sotaque in sotaques:
        df_filtered = df[df["sotaque"] == sotaque]
        X_sotaque, _ = generate_dataset_from_df(
            df_filtered, sample_duration=sample_duration)
        y_sotaque = [sotaque] * len(X_sotaque)
        X.extend(X_sotaque)
        y.extend(y_sotaque)
    X = np.array(X)
    y = np.array(y)
    return X, y


def get_treated_df():
    """
    Get the treated dataframe.

    :return: the treated dataframe
    """
    df = pd.read_csv("data/merged_information.csv")

    # Join cidade and estado
    df["cidade_estado"] = df["cidade"] + "-" + df["estado"]

    # Parse accent from cidade_estado
    map_accent = {
        "São Paulo-SP": "paulistano",
        "Brasília-DF": "brasiliense",
        "Cotia-SP": "caipira",
        "Goiânia-GO": "sertanejo",
        "Fortaleza-CE": "costa norte",
        "São Joaquim da Barra-SP": "caipira",
        "São Carlos-SP": "caipira",
        "Santo André-SP": "paulistano",
        "Barbacena-MG": "mineiro",
        "Tupã-SP": "caipira",
        "Ribeirão Bonito-SP": "caipira",
        "Colatina-ES": "fluminense",
        "Santos-SP": "paulistano",
        "Ribeirão Preto-SP": "caipira",
        "Franca-SP": "caipira",
        "Limeira-SP": "caipira",
        "Porto Feliz-SP": "caipira",
        "Bauru-SP": "caipira",
        "Piracicaba-SP": "caipira",
        "Pindamonhangaba-SP": "caipira",
        "São Caetano do Sul-SP": "paulistano",
        "São Sebastião do Paraíso-MG": "mineiro",
        "Santa Bárbara D’Oeste-SP": "caipira",
        "Vitória-ES": "fluminense",
        "Camocim de São Félix-PE": "nordestino",
        "Jundiaí-SP": "paulistano",
    }
    df["sotaque"] = df["cidade_estado"].apply(lambda x: map_accent[x])

    # Create id_frase_merge merging id_lista and id_frase
    df["id_frase_merge"] = str(df["id_lista"]) + "_" + str(df["id_frase"])

    return df


def filter_df_by_sotaques(df: pd.DataFrame, sotaques: list):
    """
    Filter a dataframe by sotaque.

    :param df: dataframe to filter
    :param sotaques: list of sotaques to filter
    :return: filtered dataframe
    """
    return df[df["sotaque"].isin(sotaques)]


def get_lpc_for_dataset(samples: np.ndarray, n_coeffs: int = 2):
    """
    Compute the LPC coefficients for a dataset.

    :param samples: audio samples
    :return: LPC coefficients
    """
    lpc_coeffs = []
    for sample in tqdm(samples):
        lpc_coeffs.append(librosa.lpc(sample, n_coeffs)[1:])
    lpc_coeffs = np.array(lpc_coeffs)
    return lpc_coeffs


def get_mfcc_for_dataset(samples: np.ndarray, sample_rate: int = 44100, n_mfcc: int = 20):
    """
    Compute the MFCC coefficients for a dataset.

    :param samples: audio samples
    :return: MFCC coefficients
    """
    mfcc = []
    for sample in tqdm(samples):
        mfcc.append(np.mean(librosa.feature.mfcc(
            y=sample, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0))
    mfcc = np.array(mfcc)
    return mfcc


def get_model_from_json(json_file: str) -> Model:
    """
    Loads a Keras model from a json file.
    :param json_file: the json file containing the model
    :return: the model
    """
    with open(json_file, 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    return model


def export_model_to_json(model: Model, json_file: str) -> None:
    """
    Exports a Keras model to a json file.
    :param model: the model
    :param json_file: the json file where the model will be exported
    :return: None
    """
    model_json = model.to_json()
    with open(json_file, "w") as json_file:
        json_file.write(model_json)


def config_dict_to_namedtuple(config: dict):
    """
    Converts a dictionary to a namedtuple.

    :param config: the dictionary
    :return: the namedtuple
    """
    return Config(**config)


def load_data(feature_sets: list, accents: list, normalize: bool = False):
    """
    Loads the data from a given feature set.

    :param feature_set: the feature set to load
    :return: X, y
    """
    df: pd.DataFrame = pd.read_csv(EXTRACTED_FEATURES_DATAFRAME)
    df = df[df["sotaque"].isin(accents)]
    X = None
    for feature_set in feature_sets:
        if X is None:
            X = df[[col for col in df.columns if col.startswith(
                feature_set)]].values
            if normalize:
                X = preprocessing.normalize(X)
        else:
            X_tmp = df[[col for col in df.columns if col.startswith(
                feature_set)]].values
            if normalize:
                X_tmp = preprocessing.normalize(X_tmp)
            X = np.concatenate((X, X_tmp), axis=1)
    y = np.array(pd.get_dummies(df["sotaque"]))
    return X, y


def store_model_weights_on_minio(model: Model):
    """
    Stores the weights of a Keras model on Minio.

    :param model: the model
    :return: None
    """
    unique_filename = str(uuid.uuid4())+".h5"
    model.save_weights(unique_filename)
    client = Minio(
        os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
    )
    client.fput_object(
        os.getenv("MINIO_BUCKET"),
        os.getenv("MINIO_PATH_PREFIX") + unique_filename,
        unique_filename
    )
    return os.getenv("MINIO_PATH_PREFIX") + unique_filename
