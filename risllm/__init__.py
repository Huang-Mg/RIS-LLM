# RIS-LLM: Reasoning-Informed Semantic Large Language Model
# for electricity market price dynamics modeling and forecasting

# Submodules (no heavy deps)
from .rfe import RFEModule, ALOESS, tricube_kernel
from .tss import TSSModule
from .granger import compute_granger_matrix
from .causal_attention import CausalBiasAttention
from .prototype import PrototypeBank, PrototypeCrossAttention
from .prompt import build_structured_prompt, load_data_description
from .framework import compute_patch_stats

# Lazy import for the main model (requires transformers)
def get_model():
    from .model import RISLLMModel
    return RISLLMModel

__all__ = [
    'RFEModule', 'ALOESS', 'tricube_kernel',
    'TSSModule', 'compute_granger_matrix', 'CausalBiasAttention',
    'PrototypeBank', 'PrototypeCrossAttention',
    'build_structured_prompt', 'load_data_description',
    'compute_patch_stats',
    'get_model',
]
