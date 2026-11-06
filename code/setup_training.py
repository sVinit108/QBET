import random
import os
import time
import math
import csv
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable
from pathlib import Path

import numpy as np

import torch
from torch.types import Device
from torch.nn.modules.loss import _Loss
import torchtext

from code.quixer_model import Quixer
from code.baseline_models import Transformer, LSTM, FNet, Linformer, Performer
from code.QBET import QBET  

from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

from datasets import disable_caching, enable_caching
disable_caching()


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    """
    Computes time elapsed in minutes and seconds when given two UNIX timestamps
    with the starting time and ending time.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def batchify_s2s(
    data: torch.Tensor,
    batch_size: int,
    window_size: int,
    pad_token_id: int,
    device: Device,
) -> torch.Tensor:
    batch_nr_of_elements = batch_size * window_size
    nr_of_batches = (data.size(0) - 1) // batch_nr_of_elements

    batched_data = (
        data[: nr_of_batches * batch_nr_of_elements]
        .view(batch_nr_of_elements, nr_of_batches)
        .T
    )

    window_data = torch.cat(
        (
            torch.full((window_size, 1), pad_token_id, device=device),
            batched_data[-window_size:, :-1],
        ),
        dim=1,
    )

    return torch.cat((window_data, batched_data))


def get_batch_s2s(
    source: torch.Tensor, i: int, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return source[i : i + window_size].T, source[i + window_size]


def initialise_weights(model: torch.nn.Module) -> None:
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)


def setup_dataset(
    device: Device, batch_size: int, window_size: int, dataset_name: str
) -> Tuple[torchtext.vocab.Vocab, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    
    print(f"[dataset] Using: {dataset_name.upper()}")

    if dataset_name.lower() == "ptb":
        raw_dset = load_dataset("ptb_text_only")
        train_sents = [s.as_py() for s in raw_dset["train"].data[0]]
        val_sents   = [s.as_py() for s in raw_dset["validation"].data[0]]
        test_sents  = [s.as_py() for s in raw_dset["test"].data[0]]
    elif dataset_name.lower() == "wiki2":
        raw_dset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_sents = [s for s in raw_dset["train"]["text"] if s.strip()]
        val_sents   = [s for s in raw_dset["validation"]["text"] if s.strip()]
        test_sents  = [s for s in raw_dset["test"]["text"] if s.strip()]
    elif dataset_name.lower() == "wiki103":
        raw_dset = load_dataset("wikitext", "wikitext-103-raw-v1")
        train_sents = [s for s in raw_dset["train"]["text"] if s.strip()]
        val_sents   = [s for s in raw_dset["validation"]["text"] if s.strip()]
        test_sents  = [s for s in raw_dset["test"]["text"] if s.strip()]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'ptb', 'wiki2', or 'wiki103'.")

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_sents), specials=["<pad>", "<unk>", "<eos>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    print(f"[dataset] Vocabulary size : {len(vocab):,}")

    def data_process(raw_text_iter) -> torch.Tensor:
        data = [
            torch.tensor(vocab(tokenizer(item)) + [vocab["eos"]], dtype=torch.long)
            for item in raw_text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 1, data))).to(device)

    train_flat = data_process(train_sents)
    val_flat = data_process(val_sents)
    test_flat = data_process(test_sents)

    print(f"[dataset] Train tokens    : {len(train_flat):,}")
    print(f"[dataset] Val   tokens    : {len(val_flat):,}")
    print(f"[dataset] Test  tokens    : {len(test_flat):,}")

    PAD_TOKEN = vocab["<pad>"]

    train_iter = batchify_s2s(train_flat, batch_size, window_size, PAD_TOKEN, device)
    val_iter = batchify_s2s(val_flat, batch_size, window_size, PAD_TOKEN, device)
    test_iter = batchify_s2s(test_flat, batch_size, window_size, PAD_TOKEN, device)

    return vocab, (train_iter, val_iter, test_iter), PAD_TOKEN


def create_model(
    hyperparams: dict[str, Any], device: Device, vocabulary_size: int
) -> torch.nn.Module:
    model_str = hyperparams["model"]
    model: torch.nn.Module
    if model_str == "Quixer":
        model = Quixer(
            n_qubits=hyperparams["qubits"],
            n_tokens=hyperparams["window"],
            qsvt_polynomial_degree=hyperparams["layers"],
            n_ansatz_layers=hyperparams["ansatz_layers"],
            vocabulary_size=vocabulary_size,
            embedding_dimension=hyperparams["dimension"],
            dropout=hyperparams["dropout"],
            batch_size=hyperparams["batch_size"],
            device=device,
        )
    elif model_str == "FNet":
        model = FNet(
            vocab_size=vocabulary_size,
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            dropout=hyperparams["dropout"],
        )
    elif model_str == "Transformer":
        model = Transformer(
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_heads=hyperparams["heads"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
        )
    elif model_str == "LSTM":
        model = LSTM(
            emb_dim=hyperparams["dimension"],
            hid_dim=hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
        )
    elif model_str == "QBET":
        model = QBET(
            cfg=hyperparams,
            vocab_size=vocabulary_size,
            d_model=hyperparams["dimension"],   
            num_heads=hyperparams["num_heads"],
            num_layers=hyperparams["num_layers"],
            ffn_dim=hyperparams["ffn_dim"],
            num_experts=hyperparams["num_experts"]
        )
    elif model_str == "Linformer":
        model = Linformer(
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_heads=hyperparams["heads"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
            seq_len=hyperparams["window"],
            k=hyperparams["k"],
        )   
    elif model_str == "Performer":
        model = Performer(
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_heads=hyperparams["heads"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
            num_features=hyperparams["num_features"],
        )
    else:
        raise ValueError(f"Unrecognized model: {model_str}")

    return model


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: _Loss,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    window_size: int,
    use_q_imp: bool = True
):
    model.train()
    epoch_loss = 0
    n_batches = iterator.shape[0] - window_size
    idxs = list(range(n_batches))
    random.shuffle(idxs)

    for ctr, batch_idx in tqdm(enumerate(idxs), total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size)
        optimizer.zero_grad()

        yhat, norm_avg = model(x)
        loss = loss_function(yhat, y)

        if use_q_imp:
            surrogate_l2 = 0.0
            for block in model.modules():
                if hasattr(block, 'q_attn_circuit') and block.q_attn_circuit is not None:
                    for param in block.q_attn_circuit.surrogate.parameters():
                        surrogate_l2 += param.pow(2).sum()
            loss = loss + 0.001 * surrogate_l2

        loss.backward()
   
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / n_batches


def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: _Loss,
    window_size: int,
) -> float:
    model.eval()
    epoch_loss = 0
    n_batches = data.shape[0] - window_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches)):
            x, y = get_batch_s2s(data, batch_idx, window_size)
            yhat, _ = model(x)
            loss = loss_function(yhat, y)
            epoch_loss += loss.item()

    return epoch_loss / n_batches


def _log_epoch_data(log_filepath: Path, data: dict):
    row = [
        data.get("dataset"), data.get("ablation"), data.get("model_name"), 
        data.get("experiment_id"), data.get("seed"), data.get("epoch"),
        f"{data.get('train_loss'):.4f}", f"{data.get('valid_loss'):.4f}",
        f"{data.get('train_ppl'):.4f}", f"{data.get('valid_ppl'):.4f}",
        "", "", "", ""
    ]
    with open(log_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _log_final_data(log_filepath: Path, data: dict):
    row = [
        data.get("dataset"), data.get("ablation"), data.get("model_name"), 
        data.get("experiment_id"), data.get("seed"), "FINAL",
        "", "", "", "",
        f"{data.get('final_val_loss'):.4f}", f"{data.get('final_val_ppl'):.4f}",
        f"{data.get('final_test_loss'):.4f}", f"{data.get('final_test_ppl'):.4f}"
    ]
    with open(log_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    train_iter: torch.Tensor,
    val_iter: torch.Tensor,
    test_iter: torch.Tensor,
    experiment_id: int,
) -> float:
    
    model_name = hyperparams['model']
    seed_val   = hyperparams['seed']
    dataset_name = hyperparams.get('dataset', 'wiki103')
    use_paqm = hyperparams.get('use_paqm', True)
    use_q_imp = hyperparams.get('use_q_imp', True)
    run_ts     = int(time.time())
    
    if use_paqm and use_q_imp:
        ablation_str = "Full_Quantum"
    elif use_paqm and not use_q_imp:
        ablation_str = "PAQM_Only"
    elif not use_paqm and use_q_imp:
        ablation_str = "Q_Imp_Only"
    else:
        ablation_str = "Classical"

    folder_path = Path("./trained_models") / f"{dataset_name}_{model_name}_{ablation_str}_seed{seed_val}_{run_ts}"
    folder_path.mkdir(exist_ok=True, parents=True)
    
    checkpoint_fpath = folder_path / f"BEST_epoch_{model_name}_{seed_val}_{run_ts}.pt"

    print(f"\n{'='*60}")
    print(f"[Checkpointing] Run folder: {folder_path}/")
    print(f"  Dataset: {dataset_name} | Ablation: {ablation_str}")
    print(f"{'='*60}\n")

    log_filepath = Path(f"./training_results_{dataset_name}.csv")
    log_header = [
        "Dataset", "Ablation", "Model Name", "Experiment No", "Seed", "Epoch",
        "Train Loss", "Val Loss", "Train PPL", "Val PPL",
        "Final Val Loss", "Final Val PPL", "Final Test Loss", "Final Test PPL"
    ]
    if not log_filepath.exists():
        with open(log_filepath, 'w', newline='') as f:
            csv.writer(f).writerow(log_header)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
    )

    scheduler = None
    if hyperparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    loss_function = torch.nn.CrossEntropyLoss()

    def _evaluate(iter: torch.Tensor):
        return evaluate(model, iter, loss_function, hyperparams["window"])

    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_iter,
            optimizer,
            loss_function,
            hyperparams["max_grad_norm"],
            scheduler,
            hyperparams["window"],
            use_q_imp=use_q_imp
        )
        valid_loss = _evaluate(val_iter)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)
            
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train ppl: {train_ppl:.2f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {valid_ppl:.2f}")

        epoch_log_data = {
            "dataset": dataset_name, "ablation": ablation_str,
            "model_name": model_name, "experiment_id": experiment_id, "seed": seed_val,
            "epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss,
            "train_ppl": train_ppl, "valid_ppl": valid_ppl
        }
        _log_epoch_data(log_filepath, epoch_log_data)

    model.load_state_dict(torch.load(checkpoint_fpath))
    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)
    
    final_val_ppl = math.exp(valid_loss)
    final_test_ppl = math.exp(test_loss)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {final_val_ppl:.2f}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {final_test_ppl:.2f}")

    final_log_data = {
        "dataset": dataset_name, "ablation": ablation_str,
        "model_name": model_name, "experiment_id": experiment_id, "seed": seed_val,
        "final_val_loss": valid_loss, "final_val_ppl": final_val_ppl,
        "final_test_loss": test_loss, "final_test_ppl": final_test_ppl
    }
    _log_final_data(log_filepath, final_log_data)

    run_registry = Path("./trained_models/run_registry.txt")
    with open(run_registry, 'a') as f:
        f.write(
            f"{dataset_name}\t{ablation_str}\t{model_name}\t{seed_val}\t"
            f"{folder_path.resolve()}\t{checkpoint_fpath.name}\n"
        )
    print(f"\n[Registry] Logged to {run_registry}")

    return test_loss


def seed(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: Device) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:

        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])
        
        dataset_name = parameterization.get("dataset", "wiki103")

        vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
            device, parameterization["batch_size"], parameterization["window"], dataset_name
        )

        model = create_model(parameterization, device, len(vocab))

        initialise_weights(model)

        model = model.to(device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

        experiment_id = parameterization.get("experiment_id", 0)
        
        valid_loss = train_cycle(
            model, parameterization, train_iter, val_iter, test_iter, experiment_id
        )

        return valid_loss

    return train_evaluate