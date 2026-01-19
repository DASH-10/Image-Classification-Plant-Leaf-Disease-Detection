"""
Bu dosyada egitim ve arama (tuning) islemlerini yapiyorum.
Kisaca: model burada ogreniyor ve en iyi ayari ariyoruz.
"""
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from joblib import parallel_backend
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from .features import build_feature_matrix, encode_labels
from .models import build_svm_pipeline


def run_hog_svm_search(
    image_paths: List,
    labels: List[str],
    image_size: int,
    hog_param_grid: List[Dict],
    svm_param_grid: Dict,
    cv: int = 3,
    n_iter: Optional[int] = None,
    n_jobs: int = 1,
    cache_dir=None,
    force_recompute: bool = False,
    use_memmap: bool = True,
    prefer_threading: bool = True,
    search_subset_size: int = 5000,
    subset_strategy: str = "stratified",
    random_state: int = 42,
    search_n_iter: int = 3,
    verbose: int = 2,
) -> Tuple[object, Dict]:
    """HOG + SVM icin en iyi ayarlari arar ve en iyi modeli dondurur.

    Not: hiz icin arama asamasinda veri alt kumesi kullanilir, final model tum veriyle fit edilir.
    """
    best_score = -1.0
    best_model = None
    best_config = {}
    best_full_X = None
    best_full_y = None
    best_class_to_idx = None
    effective_n_iter = search_n_iter if n_iter is None else n_iter

    # HOG ayarlarini tek tek deniyorum
    for hog_params in hog_param_grid:
        X, kept_labels = build_feature_matrix(
            image_paths,
            image_size,
            hog_params,
            use_color_hist=True,
            labels=labels,
            cache_dir=cache_dir,
            force_recompute=force_recompute,
            use_memmap=use_memmap,
        )
        y, class_to_idx = encode_labels(kept_labels)
        n_samples = len(y)
        subset_size = n_samples
        if search_subset_size is not None and search_subset_size > 0 and n_samples > search_subset_size:
            subset_size = min(search_subset_size, n_samples)

        if subset_size < n_samples and subset_strategy.lower() == "stratified":
            class_count = len(np.unique(y))
            if subset_size < class_count:
                subset_size = n_samples

        if subset_size >= n_samples:
            X_search = X
            y_search = y
        else:
            if subset_strategy.lower() == "stratified":
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=subset_size,
                    random_state=random_state,
                )
                train_idx, _ = next(splitter.split(np.zeros(n_samples), y))
                X_search = X[train_idx]
                y_search = y[train_idx]
            else:
                X_search = X[:subset_size]
                y_search = y[:subset_size]

        print(f"Searching on subset: {len(y_search)} of total {n_samples} samples")
        pipeline = build_svm_pipeline()
        search = RandomizedSearchCV(
            pipeline,
            svm_param_grid,
            n_iter=effective_n_iter,
            cv=cv,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        if n_jobs > 1 and prefer_threading:
            with parallel_backend("threading", n_jobs=n_jobs):
                search.fit(X_search, y_search)
        else:
            search.fit(X_search, y_search)
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_config = {
                "hog_params": hog_params,
                "svm_params": search.best_params_,
                "best_svm_params": search.best_params_,
                "cv_score": search.best_score_,
                "class_to_idx": class_to_idx,
            }
            best_full_X = X
            best_full_y = y
            best_class_to_idx = class_to_idx

    if best_full_X is None or best_full_y is None or not best_config:
        return best_model, best_config

    print(f"Refitting best model on full dataset: {len(best_full_y)} samples")
    best_model = build_svm_pipeline()
    best_model.set_params(**best_config["svm_params"])
    if n_jobs > 1 and prefer_threading:
        with parallel_backend("threading", n_jobs=n_jobs):
            best_model.fit(best_full_X, best_full_y)
    else:
        best_model.fit(best_full_X, best_full_y)

    best_config["class_to_idx"] = best_class_to_idx
    return best_model, best_config


def _compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Basit metrikleri tek yerde hesaplamak icin yazdim."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def train_model(
    model,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs: int,
    early_stopping_patience: int,
    checkpoint_path: str,
):
    """Torch modeli egitir, en iyi agirliklari kaydeder ve history dondurur."""
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Her epoch'ta once train sonra val calisiyorum
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            y_true = []
            y_pred = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                y_true.extend(labels.detach().cpu().numpy())
                y_pred.extend(preds.detach().cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            metrics = _compute_metrics(np.array(y_true), np.array(y_pred))

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(metrics["accuracy"])
                history["train_macro_f1"].append(metrics["macro_f1"])
                if scheduler is not None:
                    scheduler.step()
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(metrics["accuracy"])
                history["val_macro_f1"].append(metrics["macro_f1"])

                # En iyi val loss'u bulursam modeli kaydediyorum
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, checkpoint_path)
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Uzun sure gelisme yoksa erken durduruyorum
        if epochs_no_improve >= early_stopping_patience:
            break

    model.load_state_dict(best_model_wts)
    return model, history


def run_lr_sweep(
    model_fn,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion,
    device,
    search_space: List[Dict],
    num_epochs: int,
    checkpoint_dir: str,
    early_stopping_patience: int,
):
    """Farkli ayarlari deneyip hangisi daha iyi diye bakar (basit sweep)."""
    results = []
    for idx, cfg in enumerate(search_space):
        # Modeli her denemede sifirdan kuruyorum
        model = model_fn(**cfg.get("model_kwargs", {}))
        model = model.to(device)
        optimizer = cfg["optimizer_fn"](model)
        scheduler = cfg.get("scheduler_fn", lambda opt: None)(optimizer)
        ckpt_path = f"{checkpoint_dir}/sweep_{idx}.pt"
        _, history = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            device,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=ckpt_path,
        )
        best_val = max(history["val_macro_f1"]) if history["val_macro_f1"] else 0.0
        results.append(
            {
                "config": cfg,
                "best_val_macro_f1": best_val,
                "history": history,
                "checkpoint": ckpt_path,
            }
        )
    return results
