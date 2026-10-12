"""Every model, by name. Add a model here and it becomes available as --model <Name>."""

from .bigram_counts import BigramCounts
from .bigram_logits import BigramLogits

MODELS = {
    "BigramCounts": BigramCounts,
    "BigramLogits": BigramLogits,
}
