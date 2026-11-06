import torch
import argparse

#from quixer.old_setup_training import get_train_evaluate
#from QBET.setup_training import get_train_evaluate 
from QBET.setup import get_train_evaluate

quixer_hparams = {
    "qubits": 6,
    "layers": 2,
    "ansatz_layers": 4,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "Quixer",
    "print_iter": 50,
}


lstm_hparams = {
    "layers": 2,
    "window": 32,
    "residuals": False,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.30,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "print_iter": 50,
}


fnet_hparams = {
    "layers": 2,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "FNet",
    "print_iter": 50,
}


transformer_hparams = {
    "layers": 1,
    "heads": 1,
    "window": 32,
    "epochs": 30,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.001,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "Transformer",
    "print_iter": 50,
}

# ADD OUR MODEL HYPERPARAMETERS - MINIMAL 
QBET_hparams = {
    #MODEL ARCHITECTURE
    "dimension":128,
    "num_heads": 2,             
    "num_layers": 2,            
    "ffn_dim": 512,             
    
    # QUANTUM PARAMETERS
    "n_qubits": 3,              
    "n_quantum_layers": 2,      
    "quantum_backend": "lightning.qubit",  
    "quantum_tokens": 8,        
    "attn_qubits": 2,         
    
    # SPARSE ATTENTION PARAMETERS
    "sparse_window": 32,         
    "global_tokens": 8,          
    "max_seq_len": 64,          
    
    #TRAINING PARAMETERS
    "batch_size": 32,
    "gradient_accumulation_steps": 2,
    "use_fp16": True,
    "num_experts": 1,           # Number of experts in MoE FFN
    
    #OPTIMIZATION PARAMETERS
    "window": 32, 
    "epochs": 10,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.001,
    "lr_sched": "cos", 
    "wd": 0.0001,
    "eps": 1e-10,
    "max_grad_norm": 1.0,
    
    #MODEL & MONITORING
    "model": "QBET",  # Updated model name
    "print_iter": 25,
}

linformer_hparams = {
    "layers": 2,
    "heads": 1,
    "window": 32,
    "k": 64,
    "epochs": 10,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.001,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "Linformer",
    "print_iter": 50,
}

performer_hparams = {
    "layers": 2,
    "heads": 1,
    "window": 32,
    "num_features": 32,
    "epochs": 10,
    "restart_epochs": 30000,
    "dropout": 0.10,
    "lr": 0.001,
    "lr_sched": "cos",
    "wd": 0.0001,
    "eps": 1e-10,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "model": "Performer",
    "print_iter": 50,
}

# Embedding dimensions
classical_embedding_dimensions = [128]
quantum_embedding_dimensions = [512]
our_embedding_dimensions = [128] 


model_map = {
    "Quixer": (quixer_hparams, quantum_embedding_dimensions),
    "Transformer": (transformer_hparams, classical_embedding_dimensions),
    "LSTM": (lstm_hparams, classical_embedding_dimensions),
    "FNet": (fnet_hparams, classical_embedding_dimensions),
    "QBET": (QBET_hparams, our_embedding_dimensions),  
    "Linformer": (linformer_hparams, classical_embedding_dimensions),
    "Performer": (performer_hparams, classical_embedding_dimensions),
}
available_models = list(model_map.keys())


args = argparse.ArgumentParser(
    prog="Quixer/QBET", description="Runs the QBET model and/or classical baselines"
)
args.add_argument(
    "-m",
    "--model",
    default="QBET",
    choices=available_models,
    nargs="*",
    help="Model(s) to run.",
)
args.add_argument(
    "-d", 
    "--device", 
    default="cpu", 
    help="Device to run training on."
)
args.add_argument(
    "--dataset",
    default="ptb",
    choices=["ptb", "wiki2", "wiki103"],
    help="Dataset to train on (ptb, wiki2, wiki103)."
)
args.add_argument(
    "--ablation",
    default="full_quantum",
    choices=["full_quantum", "paqm_only", "q_imp_only", "classical"],
    help="Ablation configuration for the QBET model."
)
parsed = args.parse_args()

device_name = parsed.device
dataset_name = parsed.dataset
ablation_type = parsed.ablation
models_to_run = parsed.model if type(parsed.model) is list else [parsed.model]


use_paqm = ablation_type in ["full_quantum", "paqm_only"]
use_q_imp = ablation_type in ["full_quantum", "q_imp_only"]


torch.backends.cudnn.deterministic = True

device = torch.device(device_name)
print(f"Running on device: {device}")
print(f"Dataset: {dataset_name.upper()} | Ablation Mode: {ablation_type}")

train_evaluate = get_train_evaluate(device)

experiment_counter = 0
for model_name in models_to_run:
    hyperparameters, embedding_dimensions = model_map[model_name]
    for embedding_dimension in embedding_dimensions:
        for seed in torch.randint(high=1000000, size=(1,)).tolist():
            experiment_counter += 1
            
            hyperparameters["model"] = model_name
            hyperparameters["dimension"] = embedding_dimension
            hyperparameters["seed"] = seed
            hyperparameters["dataset"] = dataset_name
            
            if model_name == "QBET":
                hyperparameters["use_paqm"] = use_paqm
                hyperparameters["use_q_imp"] = use_q_imp

            train_evaluate(hyperparameters)

'''''
python run.py -m QBET -d cuda --dataset wiki2 --ablation full_quantum
python run.py -m QBET -d cuda --dataset ptb --ablation paqm_only
'''
