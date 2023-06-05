import bocas
from models import create_model
from loader import load, get_n_tfrecord_files
import os
import termcolor
from datetime import datetime
import tensorflow as tf
import numpy as np
from metrics import (
    get_eval_predictions,
    evaluate_varying_snrs_accs,
    evaluate_group_accs,
    evaluate_group_varying_snr_accs,
)


def get_model(config):

    # Create the model with current configuration
    model = create_model(
        config.use_se,
        config.use_act,
        config.residual,
        config.dilate,
        config.self_attention,
    )

    return model


def get_name(config):
    now = datetime.now()
    name = f"{config.use_se}_{config.use_act}_{config.residual}_{config.dilate}_"
    name += f"{config.self_attention}_{now.strftime('%m_%d_%y_%H_%M')}_{config.version}"

    return name


def run(config):
    name = get_name(config)
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))
    termcolor.cprint(
        termcolor.colored(f"Training model: {name}", "green", attrs=["bold"])
    )
    termcolor.cprint(termcolor.colored("#" * 10, "cyan"))

    # Load datasets
    n_train_files = get_n_tfrecord_files(config.data_path, split="train")
    train_ds = load(
        config.data_path,
        split="train",
        batch_size=config.batch_size,
        shuffle_len=n_train_files,
        with_snr=False,
    )

    test_ds = load(
        config.data_path, split="test", batch_size=config.batch_size, with_snr=False,
    )

    eval_ds = load(
        config.data_path, split="test", batch_size=config.batch_size, with_snr=True,
    )

    # Get the model
    model = get_model(config)

    # Compile model
    adam = tf.keras.optimizers.Adam(lr=1e-4, clipnorm=1)
    model.compile(
        adam,
        loss=["sparse_categorical_crossentropy"],
        metrics=[
            "acc",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=2, name="top_2_acc", dtype=None
            ),
        ],
    )

    # Configure callbacks
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.1, patience=12, min_lr=1e-7, mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", mode="min", patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.experimental.BackupAndRestore(
            backup_dir=os.path.join("./models", name, "backup")
        ),
    ]

    # Train model
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        callbacks=callbacks,
        epochs=config.epochs,
        verbose=config.verbose,
    )

    # Evaluate model
    metrics = model.evaluate(eval_ds, return_dict=True, verbose=config.verbose)
    eval_pred_df = get_eval_predictions(model, eval_ds)
    snr_varying_metrics = evaluate_varying_snrs_accs(eval_pred_df)
    metrics["Max SNR Accuracy"] = np.max(list(snr_varying_metrics.values()))
    group_accs = evaluate_group_accs(eval_pred_df)
    group_varying_snr_accs = evaluate_group_varying_snr_accs(eval_pred_df)

    artifacts = [
        bocas.artifacts.KerasHistory(history, name="history"),
        bocas.artifacts.Metrics(metrics, name="metrics"),
        bocas.artifacts.Metrics(snr_varying_metrics, name="snr_varying_acc"),
        bocas.artifacts.Metrics(group_accs, name="group_acc"),
        *[
            bocas.artifacts.Metrics(res, name=f"{g_name}_snr_varying_acc")
            for g_name, res in group_varying_snr_accs.items()
        ],
    ]

    return bocas.Result(name=name, config=config, artifacts=artifacts,)
