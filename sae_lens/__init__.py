# sae_lens/__init__.py
# Lazy init for public API (PEP 562)

from typing import TYPE_CHECKING
import importlib

__version__ = "3.20.0"

__all__ = [
    "SAE",
    "SAEConfig",
    "TrainingSAE",
    "TrainingSAEConfig",
    "HookedSAETransformer",
    "ActivationsStore",
    "LanguageModelSAERunnerConfig",
    "SAETrainingRunner",
    "CacheActivationsRunnerConfig",
    "CacheActivationsRunner",
    "PretokenizeRunnerConfig",
    "PretokenizeRunner",
    "pretokenize_runner",
    "run_evals",
    "upload_saes_to_huggingface",
]

# name -> (relative_module, attr_name)
_LAZY_ATTRS = {
    # analysis
    "HookedSAETransformer": (".analysis.hooked_sae_transformer", "HookedSAETransformer"),

    # runners / top-level modules
    "CacheActivationsRunner": (".cache_activations_runner", "CacheActivationsRunner"),
    "SAETrainingRunner": (".sae_training_runner", "SAETrainingRunner"),
    "PretokenizeRunner": (".pretokenize_runner", "PretokenizeRunner"),
    "pretokenize_runner": (".pretokenize_runner", "pretokenize_runner"),
    "run_evals": (".evals", "run_evals"),

    # config
    "CacheActivationsRunnerConfig": (".config", "CacheActivationsRunnerConfig"),
    "LanguageModelSAERunnerConfig": (".config", "LanguageModelSAERunnerConfig"),
    "PretokenizeRunnerConfig": (".config", "PretokenizeRunnerConfig"),

    # SAE core
    "SAE": (".sae", "SAE"),
    "SAEConfig": (".sae", "SAEConfig"),

    # training
    "ActivationsStore": (".training.activations_store", "ActivationsStore"),
    "TrainingSAE": (".training.training_sae", "TrainingSAE"),
    "TrainingSAEConfig": (".training.training_sae", "TrainingSAEConfig"),
    "upload_saes_to_huggingface": (".training.upload_saes_to_huggingface", "upload_saes_to_huggingface"),
}


def __getattr__(name: str):
    """Lazy-load public symbols on first access, then cache in module globals."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    mod = importlib.import_module(mod_name, package=__name__)  # relative import inside package
    obj = getattr(mod, attr)
    globals()[name] = obj  # cache to avoid re-import
    return obj


def __dir__():
    # show lazily-available names to dir() and IDEs
    return sorted(list(globals().keys()) + list(__all__))


# Help type checkers / IDEs without importing at runtime
if TYPE_CHECKING:
    from .analysis.hooked_sae_transformer import HookedSAETransformer
    from .cache_activations_runner import CacheActivationsRunner
    from .config import (
        CacheActivationsRunnerConfig,
        LanguageModelSAERunnerConfig,
        PretokenizeRunnerConfig,
    )
    from .evals import run_evals
    from .pretokenize_runner import PretokenizeRunner, pretokenize_runner
    from .sae import SAE, SAEConfig
    from .sae_training_runner import SAETrainingRunner
    from .training.activations_store import ActivationsStore
    from .training.training_sae import TrainingSAE, TrainingSAEConfig
    from .training.upload_saes_to_huggingface import upload_saes_to_huggingface