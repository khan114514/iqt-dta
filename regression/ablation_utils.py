from __future__ import annotations

from collections import OrderedDict


SUPPORTED_VARIANTS = OrderedDict(
    [
        (
            "baseline",
            {
                "use_interaction_prior": False,
                "use_quantity_branch": False,
                "use_decorrelation_regularizer": False,
            },
        ),
        (
            "+interaction_prior",
            {
                "use_interaction_prior": True,
                "use_quantity_branch": False,
                "use_decorrelation_regularizer": False,
            },
        ),
        (
            "+quantity_branch",
            {
                "use_interaction_prior": False,
                "use_quantity_branch": True,
                "use_decorrelation_regularizer": False,
            },
        ),
        (
            "+decorrelation_only",
            {
                "use_interaction_prior": False,
                "use_quantity_branch": False,
                "use_decorrelation_regularizer": True,
            },
        ),
        (
            "+interaction_prior+quantity_branch",
            {
                "use_interaction_prior": True,
                "use_quantity_branch": True,
                "use_decorrelation_regularizer": False,
            },
        ),
        (
            "full_model",
            {
                "use_interaction_prior": True,
                "use_quantity_branch": True,
                "use_decorrelation_regularizer": True,
            },
        ),
    ]
)

VARIANT_ALIASES = {
    "baseline": "baseline",
    "default": "baseline",
    "mgraphdta": "baseline",
    "interaction_prior": "+interaction_prior",
    "+interaction_prior": "+interaction_prior",
    "quantity_branch": "+quantity_branch",
    "+quantity_branch": "+quantity_branch",
    "decorrelation_only": "+decorrelation_only",
    "+decorrelation_only": "+decorrelation_only",
    "interaction_prior+quantity_branch": "+interaction_prior+quantity_branch",
    "+interaction_prior+quantity_branch": "+interaction_prior+quantity_branch",
    "full": "full_model",
    "full_model": "full_model",
}


def normalize_model_name(model_name: str | None) -> str:
    normalized = (model_name or "MGraphDTA").strip()
    if normalized.lower() != "mgraphdta":
        raise ValueError(
            f"Unsupported model_name '{model_name}'. "
            "This framework keeps MGraphDTA as the backbone."
        )
    return "MGraphDTA"


def normalize_variant_name(variant: str | None) -> str | None:
    if variant is None:
        return None
    key = variant.strip()
    normalized = VARIANT_ALIASES.get(key, key)
    if normalized not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant '{variant}'. "
            f"Expected one of {supported_variants()}."
        )
    return normalized


def supported_variants() -> list[str]:
    return list(SUPPORTED_VARIANTS.keys())


def finalize_ablation_args(args):
    args.model_name = normalize_model_name(getattr(args, "model_name", None))

    requested_variant = normalize_variant_name(getattr(args, "variant", None) or getattr(args, "model_variant", None))
    if requested_variant is not None:
        preset = SUPPORTED_VARIANTS[requested_variant]
        for field, expected in preset.items():
            current = bool(getattr(args, field, False))
            if current and not expected:
                raise ValueError(
                    f"Variant '{requested_variant}' requires {field}={expected}, "
                    f"but the flag was explicitly set to True."
                )
        for field, expected in preset.items():
            setattr(args, field, expected)

    resolved = {
        "use_interaction_prior": bool(getattr(args, "use_interaction_prior", False)),
        "use_quantity_branch": bool(getattr(args, "use_quantity_branch", False)),
        "use_decorrelation_regularizer": bool(getattr(args, "use_decorrelation_regularizer", False)),
    }
    for variant_name, preset in SUPPORTED_VARIANTS.items():
        if preset == resolved:
            args.variant = variant_name
            args.model_variant = variant_name
            return args

    raise ValueError(
        f"Unsupported component combination {resolved}. "
        f"Expected one of {supported_variants()}."
    )
