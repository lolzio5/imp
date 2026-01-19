import numpy as np
from fewshot.data.episode import Episode


def preprocess_batch(batch):
    """
    Ensure batch data has shape:
        x_* : [B, N, C, H, W]
        y_* : [B, N]
    """

    # Case: input comes without batch dimension
    if batch.x_train.ndim == 4:
        x_train = np.expand_dims(batch.x_train, 0)
        y_train = np.expand_dims(batch.y_train, 0)

        x_test = np.expand_dims(batch.x_test, 0)
        y_test = np.expand_dims(batch.y_test, 0)

        if batch.x_unlabel is not None:
            x_unlabel = np.expand_dims(batch.x_unlabel, 0)
            x_unlabel = np.rollaxis(x_unlabel, 4, 2)
        else:
            x_unlabel = None

        if hasattr(batch, 'y_unlabel') and batch.y_unlabel is not None:
            y_unlabel = np.expand_dims(batch.y_unlabel, 0)
        else:
            y_unlabel = None
    else:
        # Already batched
        x_train = batch.x_train
        y_train = batch.y_train
        x_test = batch.x_test
        y_test = batch.y_test
        x_unlabel = batch.x_unlabel
        y_unlabel = batch.y_unlabel if hasattr(batch, 'y_unlabel') else None

    # Convert NHWC -> NCHW
    x_train = np.rollaxis(x_train, 4, 2)
    x_test = np.rollaxis(x_test, 4, 2)

    return Episode(
        x_train,
        y_train,
        batch.train_indices,
        x_test,
        y_test,
        batch.test_indices,
        x_unlabel=x_unlabel,
        y_unlabel=y_unlabel,
        unlabel_indices=batch.unlabel_indices,
        y_train_str=batch.y_train_str,
        y_test_str=batch.y_test_str
    )
