# ==============================================================================
# AffinityLM: Drug-Target Affinity Prediction using Pretrained Language Models
# Model: MolFormer (Ligand) + ESM-2 (Protein) + XGBoost Regressor
# ==============================================================================

import os
import gc
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from xgboost.callback import EarlyStopping

# ==============================================================================
# 1. CONFIGURATION AND HYPERPARAMETERS
# ==============================================================================

# TODO: Update these paths to match your local environment
base_dir = "path/to/your/data"
DATASET_PATH = os.path.join(base_dir, "dataset.csv")
INDICES_PATH = os.path.join(base_dir, "split_indices.pkl")
MOL_EMBED_PATH = os.path.join(base_dir, "molformer_embeddings.npy")
PROT_EMBED_PATH = os.path.join(base_dir, "esm2_embeddings.npy")
OUTPUT_DIR = "results/affinity_lm/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experiment Settings
SEEDS = [1337, 123, 2024, 88, 999]
MAX_ESTIMATORS = 5000       # Maximum boosting rounds
EARLY_STOPPING_ROUNDS = 75  # Patience for early stopping

# XGBoost Hyperparameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'lambda': 2.5,  # L2 regularization
    'alpha': 0.5,   # L1 regularization
    'n_jobs': -1,
    'eval_metric': 'rmse',
    # 'device': 'cuda'  # Uncomment if GPU is available
}

# ==============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ==============================================================================

def load_data():
    """Loads embeddings and target values, verifying file integrity."""
    print("Loading pre-computed embeddings and dataset...")
    
    # Validation
    required_files = [DATASET_PATH, INDICES_PATH, MOL_EMBED_PATH, PROT_EMBED_PATH]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    # Load Feature Matrices
    X_mol = np.load(MOL_EMBED_PATH)
    X_prot = np.load(PROT_EMBED_PATH)
    
    # Feature Concatenation (Early Fusion)
    # Resulting shape: (N_samples, Mol_Dim + Prot_Dim)
    X_all = np.concatenate([X_mol, X_prot], axis=1)
    print(f"Feature matrix shape: {X_all.shape}")

    # Load Targets
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['Canonical SMILES', 'FASTA Sequence', 'target_affinity'])
    df = df[~df['target_affinity'].isin([np.inf, -np.inf])].reset_index(drop=True)
    y = df['target_affinity'].values

    # Load Split Indices
    with open(INDICES_PATH, 'rb') as f:
        indices = pickle.load(f)

    return X_all, y, indices

# ==============================================================================
# 3. MAIN EXECUTION LOOP
# ==============================================================================

if __name__ == "__main__":
    
    # Load Data
    X_all, y, indices = load_data()

    # Create Splits
    X_train, y_train = X_all[indices['train_indices']], y[indices['train_indices']]
    X_val, y_val = X_all[indices['val_indices']], y[indices['val_indices']]
    X_test, y_test = X_all[indices['test_indices']], y[indices['test_indices']]

    # Prepare Combined Train+Val for Final Training (Phase 2)
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Clean up raw arrays to save memory
    del X_all
    gc.collect()

    all_results = []

    print(f"\nStarting evaluation across {len(SEEDS)} seeds...")

    for seed in SEEDS:
        print(f"\n--- Processing Seed {seed} ---")
        
        # Update random state
        current_params = XGB_PARAMS.copy()
        current_params['random_state'] = seed

        # --- Phase 1: Determine Optimal Boosting Rounds ---
        # We train on Train set and monitor Val set to find the best 'n_estimators'
        print("Phase 1: Finding optimal boosting rounds (Early Stopping)...")
        
        es = EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)
        
        model_p1 = xgb.XGBRegressor(n_estimators=MAX_ESTIMATORS, callbacks=[es], **current_params)
        model_p1.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], 
            verbose=False
        )
        
        optimal_rounds = model_p1.best_iteration + 1
        print(f"Optimal rounds found: {optimal_rounds}")

        # --- Phase 2: Final Training ---
        # Retrain on (Train + Val) using the optimal number of rounds derived above
        print(f"Phase 2: Retraining on combined Train+Val for {optimal_rounds} rounds...")
        
        final_model = xgb.XGBRegressor(n_estimators=optimal_rounds, **current_params)
        final_model.fit(X_train_full, y_train_full, verbose=False)
        
        # Save Model Artifact
        save_path = os.path.join(OUTPUT_DIR, f"affinity_lm_seed_{seed}.json")
        final_model.save_model(save_path)

        # --- Evaluation ---
        y_pred = final_model.predict(X_test)
        
        mae = np.mean(np.abs(y_pred - y_test))
        mape = np.mean(np.abs((y_pred - y_test) / (np.abs(y_test) + 1e-10))) * 100
        pearson, _ = pearsonr(y_pred, y_test)
        ci = concordance_index(y_test, y_pred)
        
        all_results.append({'seed': seed, 'mae': mae, 'mape': mape, 'pearson': pearson, 'c_index': ci})
        print(f"Results -> MAE: {mae:.4f}, Pearson: {pearson:.4f}, C-Index: {ci:.4f}")

        # --- Plotting (Percentile Analysis) ---
        plot_path = os.path.join(OUTPUT_DIR, f"performance_seed_{seed}.png")
        
        abs_errors = np.abs(y_pred - y_test)
        sorted_indices = np.argsort(abs_errors)
        percentiles = [25, 50, 75, 90, 95, 100]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'AffinityLM Performance (Seed {seed})', fontsize=20, fontweight='bold')
        axes = axes.flatten()

        for i, perc in enumerate(percentiles):
            ax = axes[i]
            n_samples = int(len(y_test) * perc / 100)
            idxs = sorted_indices[:n_samples]
            y_p_sub, y_t_sub = y_pred[idxs], y_test[idxs]

            p_val, _ = pearsonr(y_p_sub, y_t_sub)
            ci_val = concordance_index(y_t_sub, y_p_sub)
            mae_val = np.mean(np.abs(y_p_sub - y_t_sub))

            ax.scatter(y_t_sub, y_p_sub, alpha=0.4, color='purple', s=20)
            ax.plot([y_t_sub.min(), y_t_sub.max()], [y_t_sub.min(), y_t_sub.max()], 'r--', lw=2)
            
            stats_text = (f"Pearson: {p_val:.4f}\nC-Index: {ci_val:.4f}\nMAE: {mae_val:.4f}")
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', 
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            
            ax.set_title(f'Top {perc}% ({n_samples} Samples)')
            ax.set_xlabel('Actual Affinity')
            ax.set_ylabel('Predicted Affinity')
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        
        # Cleanup
        del final_model, model_p1
        gc.collect()

    # ==============================================================================
    # 4. FINAL AGGREGATED REPORT
    # ==============================================================================
    
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("FINAL AGGREGATED RESULTS (Mean +/- SD over 5 seeds)")
    print("="*60)
    
    # Calculate Mean and Std Dev
    stats = results_df.agg(['mean', 'std'])
    
    print(f"MAE:      {stats.loc['mean', 'mae']:.4f} +/- {stats.loc['std', 'mae']:.4f}")
    print(f"MAPE:     {stats.loc['mean', 'mape']:.2f}% +/- {stats.loc['std', 'mape']:.2f}%")
    print(f"Pearson:  {stats.loc['mean', 'pearson']:.4f} +/- {stats.loc['std', 'pearson']:.4f}")
    print(f"C-Index:  {stats.loc['mean', 'c_index']:.4f} +/- {stats.loc['std', 'c_index']:.4f}")
    
    print("-" * 60)
    print("Detailed Run Results:")
    print(results_df.round(4).to_string(index=False))
    print("="*60)