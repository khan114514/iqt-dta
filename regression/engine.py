from __future__ import annotations

import hashlib
import json
import os
import platform
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

try:
    from ablation_utils import normalize_model_name
    from dataset import IndexedDataset, MergedGNNDataset
    from experiment_utils import (
        create_dir,
        get_default_data_root,
        get_default_results_root,
        load_split_indices,
        resolve_run_dir,
        save_csv,
        save_json,
        seed_everything,
        summarize_splits,
    )
    from log.basic_logger import BasicLogger
    from metrics import get_cindex, get_rm2
    from model_factory import build_model
    from utils import AverageMeter, BestMeter, load_model_dict
except ImportError:  # pragma: no cover - package-style import fallback
    from .ablation_utils import normalize_model_name
    from .dataset import IndexedDataset, MergedGNNDataset
    from .experiment_utils import (
        create_dir,
        get_default_data_root,
        get_default_results_root,
        load_split_indices,
        resolve_run_dir,
        save_csv,
        save_json,
        seed_everything,
        summarize_splits,
    )
    from .log.basic_logger import BasicLogger
    from .metrics import get_cindex, get_rm2
    from .model_factory import build_model
    from .utils import AverageMeter, BestMeter, load_model_dict


def unpack_model_output(model_output):
    if isinstance(model_output, dict):
        prediction = model_output["prediction"]
        aux_outputs = model_output.get("aux_outputs", {})
        aux_losses = model_output.get("aux_losses", {})
        return prediction, aux_outputs, aux_losses
    return model_output, {}, {}


def compute_total_loss(args, criterion, prediction, data, aux_outputs, aux_losses):
    base_loss = criterion(prediction.view(-1), data.y.view(-1))
    extra_terms = {"main_loss": base_loss}

    quantity_prediction = aux_outputs.get("quantity_prediction")
    quantity_target = getattr(data, "quantity_target", None)
    if quantity_prediction is not None and quantity_target is not None:
        quantity_loss = criterion(quantity_prediction.view(-1), quantity_target.view(-1)) * args.quantity_loss_weight
        extra_terms["quantity_branch"] = quantity_loss

    for name, value in aux_losses.items():
        extra_terms[name] = value

    total_loss = sum(extra_terms.values())
    return total_loss, extra_terms


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_rank = np.argsort(np.argsort(x)).astype(float)
    y_rank = np.argsort(np.argsort(y)).astype(float)
    return _safe_corrcoef(x_rank, y_rank)


def evaluate(model, criterion, dataloader, device, args=None, return_predictions=False):
    model.eval()
    running_loss = AverageMeter()
    running_total_loss = AverageMeter()

    pred_list = []
    label_list = []
    quantity_pred_list = []

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            model_output = model(data)
            pred, aux_outputs, aux_losses = unpack_model_output(model_output)
            if args is None:
                loss = criterion(pred.view(-1), data.y.view(-1))
                total_loss = loss
            else:
                total_loss, loss_terms = compute_total_loss(args, criterion, pred, data, aux_outputs, aux_losses)
                loss = loss_terms["main_loss"]
            labels = data.y.view(-1).detach().cpu().numpy()
            preds = pred.view(-1).detach().cpu().numpy()
            quantity_pred = aux_outputs.get("quantity_prediction")
            if quantity_pred is not None:
                quantity_pred_list.append(quantity_pred.view(-1).detach().cpu().numpy())

        pred_list.append(preds)
        label_list.append(labels)
        running_loss.update(loss.item(), len(labels))
        running_total_loss.update(total_loss.item(), len(labels))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    mse = float(np.mean((label - pred) ** 2))

    metrics = {
        "loss": float(running_loss.get_average()),
        "total_loss": float(running_total_loss.get_average()),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "ci": float(get_cindex(label, pred)),
        "pearson": _safe_corrcoef(label, pred),
        "spearman": _safe_spearman(label, pred),
        "rm2": float(get_rm2(label, pred)),
        "num_samples": int(len(label)),
    }
    if return_predictions:
        metrics["predictions"] = pred.tolist()
        metrics["labels"] = label.tolist()
        if quantity_pred_list:
            metrics["quantity_predictions"] = np.concatenate(quantity_pred_list, axis=0).tolist()
    return metrics


def get_device(device_arg=None):
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_dataloaders(args):
    dataset_root = os.path.join(args.data_root, args.dataset)
    split_indices, split_metadata, merged_df, split_audit, split_validation = load_split_indices(
        dataset_root,
        args.split_type,
        args.split_seed,
        val_fraction=args.val_fraction,
    )

    merged_dataset = MergedGNNDataset(dataset_root)
    train_set = IndexedDataset(merged_dataset, split_indices["train"])
    val_set = IndexedDataset(merged_dataset, split_indices["val"])
    test_set = IndexedDataset(merged_dataset, split_indices["test"])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    split_summary = summarize_splits(merged_df, split_indices)
    return {
        "dataset_root": dataset_root,
        "split_indices": split_indices,
        "split_metadata": split_metadata,
        "split_summary": split_summary,
        "split_audit": split_audit,
        "split_validation": split_validation,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
    }


def _save_artifacts(run_dir, args, data_bundle):
    config_payload = dict(vars(args))
    config_payload["variant"] = args.variant
    config_payload["model_name"] = normalize_model_name(args.model_name)
    save_json(os.path.join(run_dir, "config.json"), config_payload)

    artifacts_dir = os.path.join(run_dir, "artifacts")
    save_json(os.path.join(artifacts_dir, "config.json"), config_payload)
    save_json(os.path.join(artifacts_dir, "split_metadata.json"), data_bundle["split_metadata"])
    save_json(os.path.join(artifacts_dir, "split_summary.json"), data_bundle["split_summary"])
    save_json(os.path.join(artifacts_dir, "split_audit.json"), data_bundle["split_audit"])
    save_json(os.path.join(artifacts_dir, "split_validation.json"), data_bundle["split_validation"])


def _save_history(run_dir: str, history: List[Dict[str, float]]):
    save_json(os.path.join(run_dir, "artifacts", "history.json"), history)
    save_csv(os.path.join(run_dir, "artifacts", "history.csv"), history)


def _scalar_metric_payload(summary: Dict[str, object]) -> Dict[str, object]:
    metrics = summary.get("metrics", {})
    return {
        "dataset": summary.get("dataset"),
        "split": summary.get("split_type"),
        "variant": summary.get("variant") or summary.get("model_variant"),
        "seed": summary.get("train_seed"),
        "split_seed": summary.get("split_seed"),
        "best_epoch": summary.get("best_epoch"),
        "MSE": metrics.get("mse"),
        "RMSE": metrics.get("rmse"),
        "CI": metrics.get("ci"),
        "Pearson": metrics.get("pearson"),
        "Spearman": metrics.get("spearman"),
    }


def _save_metric_payload(run_dir: str, filename_prefix: str, payload: Dict[str, object]):
    save_json(os.path.join(run_dir, f"{filename_prefix}.json"), payload)
    flat_rows = []
    for key, value in payload.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_rows.append({"metric": f"{key}.{subkey}", "value": subvalue})
        else:
            flat_rows.append({"metric": key, "value": value})
    save_csv(os.path.join(run_dir, f"{filename_prefix}.csv"), flat_rows, fieldnames=["metric", "value"])


def _save_predictions(run_dir: str, split_indices: List[int], labels: List[float], predictions: List[float], quantity_predictions=None):
    rows = []
    quantity_predictions = quantity_predictions or []
    for idx, (global_index, label, prediction) in enumerate(zip(split_indices, labels, predictions)):
        row = {
            "global_index": int(global_index),
            "y_true": float(label),
            "y_pred": float(prediction),
        }
        if quantity_predictions:
            row["quantity_pred"] = float(quantity_predictions[idx])
        rows.append(row)
    save_csv(os.path.join(run_dir, "predictions.csv"), rows, fieldnames=list(rows[0].keys()) if rows else ["global_index", "y_true", "y_pred"])


def _save_reproducibility_artifact(run_dir: str, args, best_checkpoint_path: str):
    code_paths = [
        os.path.join(os.path.dirname(__file__), "ablation_utils.py"),
        os.path.join(os.path.dirname(__file__), "engine.py"),
        os.path.join(os.path.dirname(__file__), "model.py"),
        os.path.join(os.path.dirname(__file__), "model_factory.py"),
        os.path.join(os.path.dirname(__file__), "train.py"),
    ]
    digest = hashlib.sha256()
    for path in code_paths:
        digest.update(open(path, "rb").read())

    artifact = {
        "dataset": args.dataset,
        "split_type": args.split_type,
        "split_seed": args.split_seed,
        "train_seed": args.train_seed,
        "model_name": args.model_name,
        "model_variant": args.variant,
        "best_checkpoint_path": best_checkpoint_path,
        "code_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode("utf-8")).hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    save_json(os.path.join(run_dir, "artifacts", "reproducibility.json"), artifact)


def run_training(args):
    seed_everything(args.train_seed)
    device = get_device(args.device)

    create_dir(args.results_root)
    run_dir = resolve_run_dir(
        args.results_root,
        args.dataset,
        args.model_name,
        args.model_variant,
        args.split_type,
        args.split_seed,
        args.train_seed,
        run_name=args.run_name,
        overwrite=args.overwrite,
    )

    data_bundle = build_dataloaders(args)
    _save_artifacts(run_dir, args, data_bundle)

    logger = BasicLogger(os.path.join(run_dir, "train.log"))
    logger.info(f"run_dir={run_dir}")
    logger.info(f"device={device}")
    logger.info(
        f"dataset={args.dataset}, split_type={args.split_type}, split_seed={args.split_seed}, train_seed={args.train_seed}, "
        f"variant={args.variant}, "
        f"train={data_bundle['train_size']}, val={data_bundle['val_size']}, test={data_bundle['test_size']}"
    )
    logger.info(f"split_validation={data_bundle['split_validation']}")
    logger.info(f"split_audit={data_bundle['split_audit']}")

    model = build_model(
        model_name=args.model_name,
        model_variant=args.model_variant,
        block_num=3,
        vocab_protein_size=25 + 1,
        embedding_size=128,
        filter_num=32,
        out_dim=1,
        use_interaction_prior=args.use_interaction_prior,
        use_quantity_branch=args.use_quantity_branch,
        use_decorrelation_regularizer=args.use_decorrelation_regularizer,
        interaction_prior_dim=args.interaction_prior_dim,
        quantity_branch_dim=args.quantity_branch_dim,
        decorrelation_dim=args.decorrelation_dim,
        interaction_prior_weight=args.interaction_prior_weight,
        quantity_branch_weight=args.quantity_branch_weight,
        quantity_loss_weight=args.quantity_loss_weight,
        decorrelation_loss_weight=args.decorrelation_loss_weight,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]

    history = []
    running_main_loss = AverageMeter()
    running_total_loss = AverageMeter()
    running_cindex = AverageMeter()
    best_val = BestMeter("min")
    best_epoch = None
    best_step = None
    best_val_total_loss = None
    best_checkpoint_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
    break_flag = False
    global_step = 0
    validation_interval = max(0, int(args.steps_per_epoch))

    def run_validation(current_epoch: int, current_step: int):
        nonlocal best_epoch, best_step, best_val_total_loss, break_flag
        if running_main_loss.count == 0:
            return

        train_metrics = {
            "loss": float(running_main_loss.get_average()),
            "total_loss": float(running_total_loss.get_average()),
            "ci": float(running_cindex.get_average()),
        }
        running_main_loss.reset()
        running_total_loss.reset()
        running_cindex.reset()

        val_metrics = evaluate(model, criterion, val_loader, device, args=args)
        model.train()
        log_row = {
            "epoch": current_epoch,
            "global_step": current_step,
            "train_loss": train_metrics["loss"],
            "train_total_loss": train_metrics["total_loss"],
            "train_ci": train_metrics["ci"],
            "val_loss": val_metrics["loss"],
            "val_total_loss": val_metrics["total_loss"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_ci": val_metrics["ci"],
            "val_pearson": val_metrics["pearson"],
            "val_spearman": val_metrics["spearman"],
            "val_rm2": val_metrics["rm2"],
        }
        history.append(log_row)

        logger.info(
            "epoch-%d step-%d, train_loss-%.4f, train_total_loss-%.4f, train_ci-%.4f, "
            "val_loss-%.4f, val_total_loss-%.4f, val_mse-%.4f, val_ci-%.4f, val_pearson-%.4f, val_spearman-%.4f"
            % (
                current_epoch,
                current_step,
                train_metrics["loss"],
                train_metrics["total_loss"],
                train_metrics["ci"],
                val_metrics["loss"],
                val_metrics["total_loss"],
                val_metrics["mse"],
                val_metrics["ci"],
                val_metrics["pearson"],
                val_metrics["spearman"],
            )
        )

        if val_metrics["loss"] < best_val.get_best():
            best_val.update(val_metrics["loss"])
            best_epoch = current_epoch
            best_step = current_step
            best_val_total_loss = val_metrics["total_loss"]
            torch.save(model.state_dict(), best_checkpoint_path)
            _save_metric_payload(
                run_dir,
                "best_val_metrics",
                {
                    "epoch": best_epoch,
                    "global_step": best_step,
                    "dataset": args.dataset,
                    "split_type": args.split_type,
                    "split_seed": args.split_seed,
                    "train_seed": args.train_seed,
                    "variant": args.variant,
                    "model_variant": args.variant,
                    "selection_metric": "loss",
                    "metrics": val_metrics,
                },
            )
            logger.info(f"saved best checkpoint to {best_checkpoint_path}")
        else:
            best_val.counter()
            if best_epoch is not None and current_epoch - best_epoch > args.early_stop_epoch:
                logger.info(f"early stop in epoch {current_epoch} step {current_step}")
                break_flag = True

    for epoch_idx in range(1, int(args.epochs) + 1):
        if break_flag:
            break

        model.train()
        for data in train_loader:
            global_step += 1
            data = data.to(device)
            model_output = model(data)
            pred, aux_outputs, aux_losses = unpack_model_output(model_output)

            loss, loss_terms = compute_total_loss(args, criterion, pred, data, aux_outputs, aux_losses)
            cindex = get_cindex(
                data.y.detach().cpu().numpy().reshape(-1),
                pred.detach().cpu().numpy().reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_main_loss.update(loss_terms["main_loss"].item(), data.y.size(0))
            running_total_loss.update(loss.item(), data.y.size(0))
            running_cindex.update(cindex, data.y.size(0))

            if validation_interval > 0 and global_step % validation_interval == 0:
                run_validation(epoch_idx, global_step)
                if break_flag:
                    break

        if break_flag:
            break

        if running_main_loss.count > 0:
            run_validation(epoch_idx, global_step)

    if best_epoch is None:
        raise RuntimeError("Training finished without producing a best validation checkpoint.")

    _save_history(run_dir, history)
    _save_reproducibility_artifact(run_dir, args, best_checkpoint_path)

    load_model_dict(model, best_checkpoint_path)
    val_metrics = evaluate(model, criterion, val_loader, device, args=args)
    test_metrics_full = evaluate(model, criterion, test_loader, device, args=args, return_predictions=True)
    test_metrics = {key: value for key, value in test_metrics_full.items() if key not in {"labels", "predictions", "quantity_predictions"}}
    summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "variant": args.variant,
        "model_variant": args.variant,
        "split_type": args.split_type,
        "split_seed": args.split_seed,
        "train_seed": args.train_seed,
        "run_dir": run_dir,
        "best_checkpoint_path": best_checkpoint_path,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_val_loss": float(best_val.get_best()),
        "best_val_total_loss": None if best_val_total_loss is None else float(best_val_total_loss),
        "metrics": test_metrics,
    }
    _save_metric_payload(
        run_dir,
        "val_metrics",
        {
            "dataset": args.dataset,
            "split_type": args.split_type,
            "split_seed": args.split_seed,
            "train_seed": args.train_seed,
            "variant": args.variant,
            "model_variant": args.variant,
            "epoch": best_epoch,
            "global_step": best_step,
            "metrics": val_metrics,
        },
    )
    _save_metric_payload(run_dir, "test_metrics", summary)
    save_json(os.path.join(run_dir, "summary.json"), summary)
    save_json(os.path.join(run_dir, "metrics.json"), _scalar_metric_payload(summary))
    _save_predictions(
        run_dir,
        data_bundle["split_indices"]["test"],
        test_metrics_full["labels"],
        test_metrics_full["predictions"],
        quantity_predictions=test_metrics_full.get("quantity_predictions"),
    )

    logger.info(
        "final_test, best_epoch-%d, mse-%.4f, rmse-%.4f, ci-%.4f, pearson-%.4f, spearman-%.4f, rm2-%.4f"
        % (
            best_epoch,
            test_metrics["mse"],
            test_metrics["rmse"],
            test_metrics["ci"],
            test_metrics["pearson"],
            test_metrics["spearman"],
            test_metrics["rm2"],
        )
    )
    print(
        "run_dir:%s\nbest_epoch:%d\nmse:%.4f, rmse:%.4f, ci:%.4f, pearson:%.4f, spearman:%.4f, rm2:%.4f"
        % (
            run_dir,
            best_epoch,
            test_metrics["mse"],
            test_metrics["rmse"],
            test_metrics["ci"],
            test_metrics["pearson"],
            test_metrics["spearman"],
            test_metrics["rm2"],
        )
    )

    return summary


def resolve_test_loader(dataset, split_type, split_seed, data_root, batch_size, num_workers):
    dataset_root = os.path.join(data_root, dataset)
    split_indices, split_metadata, merged_df, split_audit, split_validation = load_split_indices(dataset_root, split_type, split_seed)
    merged_dataset = MergedGNNDataset(dataset_root)
    test_set = IndexedDataset(merged_dataset, split_indices["test"])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    split_summary = summarize_splits(merged_df, split_indices)

    return {
        "dataset_root": dataset_root,
        "split_indices": split_indices,
        "split_metadata": split_metadata,
        "split_summary": split_summary,
        "split_audit": split_audit,
        "split_validation": split_validation,
        "test_loader": test_loader,
        "test_size": len(test_set),
    }


def run_test(args):
    device = get_device(args.device)
    criterion = nn.MSELoss()

    model = build_model(
        model_name=args.model_name,
        model_variant=args.variant,
        block_num=3,
        vocab_protein_size=25 + 1,
        embedding_size=128,
        filter_num=32,
        out_dim=1,
        use_interaction_prior=args.use_interaction_prior,
        use_quantity_branch=args.use_quantity_branch,
        use_decorrelation_regularizer=args.use_decorrelation_regularizer,
        interaction_prior_dim=args.interaction_prior_dim,
        quantity_branch_dim=args.quantity_branch_dim,
        decorrelation_dim=args.decorrelation_dim,
        interaction_prior_weight=args.interaction_prior_weight,
        quantity_branch_weight=args.quantity_branch_weight,
        quantity_loss_weight=args.quantity_loss_weight,
        decorrelation_loss_weight=args.decorrelation_loss_weight,
    ).to(device)

    load_model_dict(model, args.model_path)
    bundle = resolve_test_loader(
        dataset=args.dataset,
        split_type=args.split_type,
        split_seed=args.split_seed,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    metrics = evaluate(model, criterion, bundle["test_loader"], device, args=args)
    print("Number of test: ", bundle["test_size"])
    print(
        "test_mse:%.4f, test_rmse:%.4f, test_ci:%.4f, test_pearson:%.4f, test_spearman:%.4f, test_rm2:%.4f"
        % (
            metrics["mse"],
            metrics["rmse"],
            metrics["ci"],
            metrics["pearson"],
            metrics["spearman"],
            metrics["rm2"],
        )
    )
    return metrics


def fill_default_paths(args):
    if not args.data_root:
        args.data_root = get_default_data_root()
    if not hasattr(args, "results_root") or not args.results_root:
        args.results_root = get_default_results_root()
    base_seed = getattr(args, "seed", None)
    if base_seed is None:
        base_seed = 0
    if not hasattr(args, "split_seed") or args.split_seed is None:
        args.split_seed = base_seed
    if not hasattr(args, "train_seed") or args.train_seed is None:
        args.train_seed = base_seed
    return args
