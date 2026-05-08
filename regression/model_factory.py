try:
    from ablation_utils import normalize_model_name, normalize_variant_name
    from model import MGraphDTA
except ImportError:  # pragma: no cover - package-style import fallback
    from .ablation_utils import normalize_model_name, normalize_variant_name
    from .model import MGraphDTA


def build_model(model_name="MGraphDTA", model_variant="baseline", **model_kwargs):
    normalize_model_name(model_name)
    normalized_variant = normalize_variant_name(model_variant) or "baseline"

    config = dict(model_kwargs)
    config.pop("model_name", None)
    config["use_interaction_prior"] = normalized_variant in {"+interaction_prior", "+interaction_prior+quantity_branch", "full_model"}
    config["use_quantity_branch"] = normalized_variant in {"+quantity_branch", "+interaction_prior+quantity_branch", "full_model"}
    config["use_decorrelation_regularizer"] = normalized_variant in {"+decorrelation_only", "full_model"}
    return MGraphDTA(**config)
