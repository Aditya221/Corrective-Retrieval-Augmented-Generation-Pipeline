from .base import EvaluatorBase, EvalResult
from .cross_encoder import CrossEncoderEvaluator
from .llm_judge import LLMEvaluator


def get_evaluator(backend: str = "cross_encoder", **kwargs) -> EvaluatorBase:
    backends = {
        "cross_encoder": CrossEncoderEvaluator,
        "llm": LLMEvaluator,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(backends.keys())}")
    return backends[backend](**kwargs)


__all__ = ["get_evaluator", "EvaluatorBase", "EvalResult"]
