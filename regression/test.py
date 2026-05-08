import argparse
import json
import os

try:
    from ablation_utils import finalize_ablation_args
    from engine import fill_default_paths, run_test
    from experiment_utils import SPLIT_TYPES
except ImportError:  # pragma: no cover - package-style import fallback
    from .ablation_utils import finalize_ablation_args
    from .engine import fill_default_paths, run_test
    from .experiment_utils import SPLIT_TYPES


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate an MGraphDTA checkpoint on a split-aware regression test set."
    )
    parser.add_argument("--dataset", required=False, default=None, help="Dataset name, for example davis or kiba.")
    parser.add_argument("--model_path", required=False, type=str, help="Path to a checkpoint or state_dict.")
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Optional experiment run directory. If set, dataset/model_path/split_type/seed can be inferred.",
    )
    parser.add_argument("--split_type", default=None, choices=SPLIT_TYPES, help="Split protocol used for testing.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used by the split definition.")
    parser.add_argument("--split_seed", type=int, default=None, help="Explicit split seed. Defaults to --seed.")
    parser.add_argument("--batch_size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--model_name", default=None, help="Backbone name. Must remain MGraphDTA.")
    parser.add_argument("--variant", default=None, help="Optional ablation preset name.")
    parser.add_argument("--model_variant", default=None, help="Compatibility alias for --variant.")
    parser.add_argument("--use_interaction_prior", action="store_true", help="Enable component A: interaction_prior.")
    parser.add_argument("--use_quantity_branch", action="store_true", help="Enable component B: quantity_branch.")
    parser.add_argument("--use_decorrelation_regularizer", action="store_true", help="Enable component C: decorrelation_regularizer.")
    parser.add_argument("--quantity_loss_weight", type=float, default=0.2, help="Auxiliary loss weight for quantity_branch.")
    parser.add_argument("--decorrelation_loss_weight", type=float, default=1e-2, help="Regularization weight for decorrelation_regularizer.")
    parser.add_argument("--interaction_prior_dim", type=int, default=128, help="Hidden dimension for interaction_prior.")
    parser.add_argument("--quantity_branch_dim", type=int, default=128, help="Hidden dimension for quantity_branch.")
    parser.add_argument("--decorrelation_dim", type=int, default=128, help="Hidden dimension for decorrelation_regularizer.")
    parser.add_argument(
        "--interaction_prior_weight",
        type=float,
        default=1.0,
        help="Feature scaling weight applied to interaction_prior before fusion.",
    )
    parser.add_argument(
        "--quantity_branch_weight",
        type=float,
        default=1.0,
        help="Feature scaling weight applied to quantity_branch hidden features before fusion.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--data_root", default=None, help="Root directory containing regression datasets.")
    parser.add_argument("--results_root", default=None, help="Root directory for structured experiment outputs.")
    return parser


def hydrate_args_from_run_dir(args):
    if not args.run_dir:
        if not args.dataset or not args.model_path:
            raise ValueError("Either provide --run_dir or provide both --dataset and --model_path.")
        if args.split_type is None:
            args.split_type = "random"
        if args.seed is None:
            args.seed = 0
        if args.model_name is None:
            args.model_name = "MGraphDTA"
        if args.model_variant is None:
            args.model_variant = "baseline"
        return args

    config_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.run_dir, "artifacts", "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        args.dataset = args.dataset or config.get("dataset")
        args.split_type = args.split_type or config.get("split_type", "random")
        args.seed = config.get("seed", 0) if args.seed is None else args.seed
        args.split_seed = config.get("split_seed", args.seed) if args.split_seed is None else args.split_seed
        args.model_name = args.model_name or config.get("model_name", "MGraphDTA")
        args.variant = args.variant or config.get("variant") or config.get("model_variant")
        args.model_variant = args.model_variant or args.variant
        args.use_interaction_prior = bool(config.get("use_interaction_prior", args.use_interaction_prior))
        args.use_quantity_branch = bool(config.get("use_quantity_branch", args.use_quantity_branch))
        args.use_decorrelation_regularizer = bool(
            config.get("use_decorrelation_regularizer", args.use_decorrelation_regularizer)
        )
        args.quantity_loss_weight = config.get("quantity_loss_weight", args.quantity_loss_weight)
        args.decorrelation_loss_weight = config.get("decorrelation_loss_weight", args.decorrelation_loss_weight)
        args.interaction_prior_dim = config.get("interaction_prior_dim", args.interaction_prior_dim)
        args.quantity_branch_dim = config.get("quantity_branch_dim", args.quantity_branch_dim)
        args.decorrelation_dim = config.get("decorrelation_dim", args.decorrelation_dim)
        args.interaction_prior_weight = config.get("interaction_prior_weight", args.interaction_prior_weight)
        args.quantity_branch_weight = config.get("quantity_branch_weight", args.quantity_branch_weight)

    if not args.model_path:
        args.model_path = os.path.join(args.run_dir, "checkpoints", "best_model.pt")
    if not args.dataset:
        raise ValueError(f"Unable to infer --dataset from run directory {args.run_dir}.")
    if args.split_type is None:
        args.split_type = "random"
    if args.seed is None:
        args.seed = 0
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.model_name is None:
        args.model_name = "MGraphDTA"
    if args.variant is None and args.model_variant is not None:
        args.variant = args.model_variant
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.variant and args.model_variant and args.variant != args.model_variant:
        raise ValueError(f"--variant ({args.variant}) and --model_variant ({args.model_variant}) must match.")
    if args.variant is None and args.model_variant is not None:
        args.variant = args.model_variant
    fill_default_paths(args)
    hydrate_args_from_run_dir(args)
    finalize_ablation_args(args)
    run_test(args)


if __name__ == "__main__":
    main()
