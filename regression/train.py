import argparse

try:
    from ablation_utils import finalize_ablation_args, supported_variants
    from engine import fill_default_paths, run_training
    from experiment_utils import SPLIT_TYPES
except ImportError:  # pragma: no cover - package-style import fallback
    from .ablation_utils import finalize_ablation_args, supported_variants
    from .engine import fill_default_paths, run_training
    from .experiment_utils import SPLIT_TYPES


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extensible MGraphDTA training entrypoint for regression experiments."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, for example davis or kiba.")
    parser.add_argument(
        "--split_type",
        default="random",
        choices=SPLIT_TYPES,
        help="Split protocol. Non-random splits expect pre-generated manifests under data/{dataset}/splits/{split_type}/.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Compatibility seed. Used for both split_seed and train_seed if not set.")
    parser.add_argument("--split_seed", type=int, default=None, help="Seed for split generation / manifest selection.")
    parser.add_argument("--train_seed", type=int, default=None, help="Seed for model initialization and dataloader shuffling.")
    parser.add_argument("--save_model", action="store_true", help="Compatibility flag. Best checkpoint is always saved.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=3000, help="Number of full passes over the training loader.")
    parser.add_argument("--steps_per_epoch", type=int, default=50, help="Validation cadence in optimizer steps.")
    parser.add_argument("--early_stop_epoch", type=int, default=400, help="Patience on validation loss.")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--val_fraction", type=float, default=0.1, help="Validation fraction for legacy random split fallback.")
    parser.add_argument("--model_name", default="MGraphDTA", help="Backbone name. Must remain MGraphDTA.")
    parser.add_argument(
        "--variant",
        default=None,
        help=f"Optional ablation preset. Supported values: {supported_variants()}",
    )
    parser.add_argument(
        "--model_variant",
        default=None,
        help="Compatibility alias for --variant. If both are set they must agree.",
    )
    parser.add_argument("--use_interaction_prior", action="store_true", help="Enable component A: interaction_prior.")
    parser.add_argument("--use_quantity_branch", action="store_true", help="Enable component B: quantity_branch.")
    parser.add_argument(
        "--use_decorrelation_regularizer",
        action="store_true",
        help="Enable component C: decorrelation_regularizer.",
    )
    parser.add_argument("--quantity_loss_weight", type=float, default=0.2, help="Auxiliary loss weight for quantity_branch.")
    parser.add_argument(
        "--decorrelation_loss_weight",
        type=float,
        default=1e-2,
        help="Regularization weight for decorrelation_regularizer.",
    )
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
    parser.add_argument("--run_name", default=None, help="Optional stable run name inside the seed directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an explicit run_name directory.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.variant and args.model_variant:
        if args.variant != args.model_variant:
            raise ValueError(f"--variant ({args.variant}) and --model_variant ({args.model_variant}) must match.")
    if args.variant is None and args.model_variant is not None:
        args.variant = args.model_variant
    fill_default_paths(args)
    finalize_ablation_args(args)
    run_training(args)


if __name__ == "__main__":
    main()
