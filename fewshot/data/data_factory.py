import os


DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
  """Registers a dataset class"""

  def decorator(f):
    DATASET_REGISTRY[dataset_name] = f
    return f

  return decorator


def get_data_folder(data_root, dataset_name):
  # If no data_root provided, default to the repository root (two levels up from this file).
  if data_root is None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_root = repo_root

  data_folder = os.path.join(data_root, dataset_name)
  return data_folder


def get_dataset(FLAGS, dataset_name, split, *args, **kwargs):
  if dataset_name in DATASET_REGISTRY:
    return DATASET_REGISTRY[dataset_name](FLAGS, get_data_folder(FLAGS.data_root, dataset_name), split,
                                          *args, **kwargs)
  else:
    raise ValueError("Unknown dataset \"{}\"".format(dataset_name))
