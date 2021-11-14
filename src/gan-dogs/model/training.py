import time

from .callbacks import get_callbacks


def train(gan, train_dataset, n_epochs, epochs_so_far=0, checkpoint_path=None):
    """Train the GAN model

    Args:
        gan (GAN): GAN model to train
        train_dataset (tf.data.Dataset): training dataset
        n_epochs (int): number of epochs to train
        epochs_so_far (int, optional): starting epoch (in case of resuming training). Defaults to 0.
        checkpoint_path (str, optional): path to checkpoint for saving model training progress. Defaults to None.

    Returns:
        tf.keras.callbacks.History: training history
    """
    # Sanity check
    batch = next(iter(train_dataset))
    s = time.time()
    gan.train_step(batch)
    e = time.time()
    print(f"Time for step: {e-s} s")

    callbacks = get_callbacks(train_dataset, checkpoint_path)

    try:
        history = gan.fit(
            train_dataset,
            epochs=n_epochs,
            initial_epoch=epochs_so_far,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("\n\n")
        print("Training interrupted by user.")

    return history
