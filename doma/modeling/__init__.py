"""
Training and evaluation for gesture models.

- train: generic training script (model-agnostic) with metrics and progress.
"""

from doma.modeling.train import run_train, get_model_builder, register_model

__all__ = ["run_train", "get_model_builder", "register_model"]
