import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold

from db import *
from utils import *


def train_with_config(config_file_path: str, model_json_path: str):
    """
    Train a model using the configuration file.
    :param config_file_path: path to the configuration file
    :param model_json_path: path to the model json file
    """
    # Load the configuration file
    with open(config_file_path) as f:
        config_dict = json.load(f)
        config: Config = config_dict_to_namedtuple(config_dict)

    # Initialize DBManager
    db_manager: DBManager = DBManager()

    # Load data
    X, y = load_data(config.feature_sets, normalize=True)

    # Setup cross-validation
    kf = KFold(n_splits=config.n_folds, shuffle=True,
               random_state=config.random_seed)

    # Initialize scores
    scores = []

    # Loop over folds
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f'Fold {i}')

        # Clear the session
        K.clear_session()

        # Split the data
        X_train, X_test = X[train_index], X[test_index]

        # Build model
        model: Model = get_model_from_json(model_json_path)
        model.compile(optimizer=config.optimizer,
                      loss=config.loss_func,
                      metrics=["AUC"])

        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.monitor,
            factor=config.lr_factor,
            patience=config.lr_patience,
            verbose=config.verbose,
        )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=config.monitor,
            patience=config.es_patience,
            verbose=config.verbose,
        )

        # Train the model
        history = model.fit(
            X_train,
            y[train_index],
            epochs=config.max_epochs,
            batch_size=config.batch_size,
            validation_data=(X_test, y[test_index]),
            callbacks=[reduce_lr, early_stopping],
            verbose=config.verbose,
        )
        scores.append(history.history['val_AUC'][-1])
        print(f'AUC: {scores[-1]}')
        print()

    # Print the average AUC
    val_auc = np.mean(scores)
    print(f'Average AUC: {val_auc}')

    # Train with full data for saving weights
    model: Model = get_model_from_json(model_json_path)
    model.compile(
        optimizer=config.optimizer,
        loss=config.loss_func,
        metrics=["AUC"],
    )
    history = model.fit(
        X,
        y,
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=[reduce_lr, early_stopping],
        verbose=config.verbose,
    )

    # Save weights on MinIO
    weights_fname = store_model_weights_on_minio(model)

    # Add results to database
    db_manager.add_result(
        config=config_dict,
        val_aux=val_auc,
        weights_path=weights_fname,
    )
