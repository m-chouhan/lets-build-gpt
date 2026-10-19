"""How a model's numbers get set from data. Each model names its method via `train_method`.

Counting is closed-form: one exact pass over the data.
"""


def train_by_counting(model, train_data, val_data, cfg):
    model.count_pairs(train_data)


TRAIN_METHODS = {
    "counting": train_by_counting,
}


def train(model, train_data, val_data, cfg):
    """Train the model the way it says it learns (model.train_method)."""
    TRAIN_METHODS[model.train_method](model, train_data, val_data, cfg)
