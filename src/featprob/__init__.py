"""featprob: probabilistic modelling of deep features."""

from featprob.gmm import ClassConditionalGMM, evaluate_ood

__all__ = ["ClassConditionalGMM", "evaluate_ood", "__version__"]
__version__ = "0.1.0"
