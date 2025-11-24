# ==============================================================================
# AffinityNet: A Deep Learning Model for Drug-Target Affinity Prediction
# ==============================================================================

import os
import gc
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, LayerNormalization, 
    MultiHeadAttention, Add, Layer, Embedding, 
    GlobalAveragePooling1D, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.regularizers import L2
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

# --- GPU & MIXED PRECISION SETUP ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# TODO: Update these paths to match your local environment
DATASET_PATH = 'path/to/your/dataset.csv' 
INDICES_PATH = 'path/to/your/COLD_SPLIT_INDICES.pkl'
OUTPUT_DIR = 'results/'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Experiment Settings
SEEDS = [1337, 123, 2024, 88, 999]
BATCH_SIZE = 16
MAX_EPOCHS = 150

# Data Dimensions
MAX_LIG_LEN = 240
MAX_PROT_LEN = 960

# Model Hyperparameters
LIG_EMBED_DIM = 48
PROT_EMBED_DIM = 64
FF_DIM = 256        # Transformer Feed-Forward Dimension
PROJ_DIM = 96       # Projection Dimension before pooling
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-4

# ==============================================================================
# CUSTOM LAYERS & UTILS
# ==============================================================================

def huber_loss(y_true, y_pred, delta=3.0):
    """Robust regression loss function."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * tf.abs(error) - 0.5 * tf.square(delta)
    return tf.where(is_small_error, squared_loss, linear_loss)

class CustomLeakyReLU(Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        return tf.nn.leaky_relu(inputs, alpha=self.alpha)

class GraphAttentionWithAdjacency(Layer):
    """
    Implements Graph Attention logic where adjacency is pre-calculated 
    based on sequence locality (1st and 2nd neighbors).
    """
    def __init__(self, units, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.seq_len = seq_len
        
        # Pre-compute adjacency matrix for local structure (neighbors +/- 1 and +/- 2)
        adj = np.zeros((self.seq_len, self.seq_len), dtype=np.float32)
        dist = np.abs(np.subtract.outer(np.arange(self.seq_len), np.arange(self.seq_len)))
        adj[dist <= 1] = 1.0  # Immediate neighbors
        adj[dist == 2] = 0.5  # Second-order neighbors
        self.adjacency_matrix = tf.constant(adj, dtype=tf.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.units), 
                                      initializer='glorot_uniform', regularizer=L2(1e-3))
        self.attn_kernel = self.add_weight(name='attn_kernel', shape=(self.units * 2, 1), 
                                           initializer='glorot_uniform', regularizer=L2(1e-3))

    def call(self, inputs):
        node_features = tf.matmul(inputs, self.kernel)
        
        # Prepare for attention mechanism (broadcasting)
        tiled = tf.tile(node_features[:, :, tf.newaxis, :], [1, 1, self.seq_len, 1])
        tiled_trans = tf.transpose(tiled, [0, 2, 1, 3])
        concat = Concatenate(axis=-1)([tiled, tiled_trans])
        
        scores = tf.squeeze(tf.matmul(concat, self.attn_kernel), axis=-1)
        scores = tf.nn.leaky_relu(scores, alpha=0.2)
        
        # Apply adjacency mask
        adjacency_matrix_casted = tf.cast(self.adjacency_matrix, dtype=scores.dtype)
        mask = tf.cast(tf.math.not_equal(adjacency_matrix_casted, 0.0), dtype=scores.dtype)
        mask_value = np.finfo(scores.dtype.as_numpy_dtype).min
        
        weighted_scores = scores * adjacency_matrix_casted
        final_scores = weighted_scores + (1.0 - mask) * mask_value
        attn_weights = tf.nn.softmax(final_scores, axis=-1)
        
        return tf.matmul(attn_weights, node_features)

def transformer_encoder(inputs, num_heads_list, ff_dim):
    """Stack of Transformer Encoder blocks."""
    x = inputs
    for num_heads in num_heads_list:
        # Attention Sub-layer
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=max(1, x.shape[-1] // num_heads))(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # Feed-Forward Sub-layer
        ffn = Dense(ff_dim, activation=CustomLeakyReLU(), kernel_regularizer=L2(1e-3))(x)
        ffn = Dense(x.shape[-1], kernel_regularizer=L2(1e-3))(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization()(x)
    return x

def encode_sequence(seq, char_to_int, max_len):
    encoded = [char_to_int.get(char, 0) for char in seq]
    return tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post')[0]

# ==============================================================================
# MODEL BUILDER
# ==============================================================================

def build_keras_model(smi_vocab_size, pro_vocab_size):
    # Inputs
    lig_input = Input(shape=(MAX_LIG_LEN,), name='lig_input', dtype='int32')
    prot_input = Input(shape=(MAX_PROT_LEN,), name='prot_input', dtype='int32')
    
    # Embeddings
    lig_embed = Embedding(smi_vocab_size, LIG_EMBED_DIM)(lig_input)
    prot_embed = Embedding(pro_vocab_size, PROT_EMBED_DIM)(prot_input)
    
    # Structural Encoding (GAT)
    lig_gat_out = GraphAttentionWithAdjacency(LIG_EMBED_DIM, MAX_LIG_LEN)(lig_embed)
    prot_gat_out = GraphAttentionWithAdjacency(PROT_EMBED_DIM, MAX_PROT_LEN)(prot_embed)
    
    # Contextual Encoding (Transformer)
    lig_trans_out = transformer_encoder(lig_gat_out, [2, 4], FF_DIM)
    prot_trans_out = transformer_encoder(prot_gat_out, [2, 4, 6], FF_DIM)
    
    # Projection & Pooling
    lig_proj = Dense(PROJ_DIM, activation=CustomLeakyReLU())(lig_trans_out)
    prot_proj = Dense(PROJ_DIM, activation=CustomLeakyReLU())(prot_trans_out)
    
    lig_pool = Concatenate()([GlobalAveragePooling1D()(lig_proj), GlobalMaxPooling1D()(lig_proj)])
    prot_pool = Concatenate()([GlobalAveragePooling1D()(prot_proj), GlobalMaxPooling1D()(prot_proj)])
    
    # Interaction Vector
    concatenated = Concatenate(name='concat_lig_prot')([lig_pool, prot_pool])
    
    # Prediction Head (Auxiliary)
    x = Dense(512, activation=CustomLeakyReLU(), kernel_regularizer=L2(1e-3))(concatenated)
    x = Dropout(DROPOUT_RATE)(x)
    x = Dense(256, activation=CustomLeakyReLU(), kernel_regularizer=L2(1e-3))(x)
    x = Dropout(DROPOUT_RATE)(x)
    output = Dense(1, activation='linear', dtype='float32')(x)
    
    model = Model(inputs=[lig_input, prot_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=huber_loss)
    return model

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Load and Preprocess Data
    print("Loading data...")
    if not os.path.exists(DATASET_PATH) or not os.path.exists(INDICES_PATH):
        raise FileNotFoundError(f"Please check paths for DATASET_PATH and INDICES_PATH in configuration.")

    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['Canonical SMILES', 'FASTA Sequence', 'target_affinity'])
    df = df[~df['target_affinity'].isin([np.inf, -np.inf])].reset_index(drop=True)
    
    y = df['target_affinity'].values
    smiles_list = df['Canonical SMILES'].tolist()
    fasta_list = df['FASTA Sequence'].tolist()

    # Build Vocabulary
    smiles_chars = sorted(list(set(''.join(smiles_list))))
    protein_chars = sorted(list(set(''.join(fasta_list))))
    smi_char_to_int = {char: idx + 1 for idx, char in enumerate(smiles_chars)}
    pro_char_to_int = {char: idx + 1 for idx, char in enumerate(protein_chars)}
    
    SMI_VOCAB_SIZE = len(smi_char_to_int) + 1
    PRO_VOCAB_SIZE = len(pro_char_to_int) + 1

    # Encode Inputs
    print("Encoding sequences...")
    X_smiles = np.array([encode_sequence(s, smi_char_to_int, MAX_LIG_LEN) for s in smiles_list])
    X_fasta = np.array([encode_sequence(f, pro_char_to_int, MAX_PROT_LEN) for f in fasta_list])

    with open(INDICES_PATH, 'rb') as f:
        indices = pickle.load(f)

    # Split Data
    X_lig_train, X_lig_val, X_lig_test = X_smiles[indices['train_indices']], X_smiles[indices['val_indices']], X_smiles[indices['test_indices']]
    X_prot_train, X_prot_val, X_prot_test = X_fasta[indices['train_indices']], X_fasta[indices['val_indices']], X_fasta[indices['test_indices']]
    y_train, y_val, y_test = y[indices['train_indices']], y[indices['val_indices']], y[indices['test_indices']]

    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    all_results = []

    # 2. Multi-Seed Training Loop
    for seed in SEEDS:
        print(f"\n--- Starting Run for Seed: {seed} ---")
        
        # Reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Define paths
        CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"checkpoint_seed_{seed}.weights.h5")
        FINAL_KERAS_PATH = os.path.join(OUTPUT_DIR, f"feature_extractor_seed_{seed}.h5")
        FINAL_XGB_PATH = os.path.join(OUTPUT_DIR, f"xgb_model_seed_{seed}.json")
        PLOT_PATH = os.path.join(OUTPUT_DIR, f"performance_plot_seed_{seed}.png")

        # --- Phase 1: Optimal Epoch Search ---
        print("Phase 1: Searching for optimal epochs...")
        phase1_model = build_keras_model(SMI_VOCAB_SIZE, PRO_VOCAB_SIZE)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ({'lig_input': X_lig_train, 'prot_input': X_prot_train}, y_train_scaled)
        ).cache().shuffle(len(X_lig_train), seed=seed).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            ({'lig_input': X_lig_val, 'prot_input': X_prot_val}, y_val_scaled)
        ).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=0),
            ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
        ]

        history = phase1_model.fit(train_dataset, epochs=MAX_EPOCHS, validation_data=val_dataset, callbacks=callbacks, verbose=1)
        optimal_epochs = np.argmin(history.history['val_loss']) + 1
        print(f"Optimal epochs found: {optimal_epochs}")
        
        # Cleanup Phase 1
        del phase1_model, history, train_dataset, val_dataset
        gc.collect()
        tf.keras.backend.clear_session()

        # --- Phase 2: Feature Extractor Training ---
        print(f"Phase 2: Training final extractor on Train+Val for {optimal_epochs} epochs...")
        final_keras_model = build_keras_model(SMI_VOCAB_SIZE, PRO_VOCAB_SIZE)
        
        X_lig_full = np.concatenate((X_lig_train, X_lig_val))
        X_prot_full = np.concatenate((X_prot_train, X_prot_val))
        y_full_scaled = np.concatenate((y_train_scaled, y_val_scaled))
        
        train_full_ds = tf.data.Dataset.from_tensor_slices(
            ({'lig_input': X_lig_full, 'prot_input': X_prot_full}, y_full_scaled)
        ).cache().shuffle(len(X_lig_full), seed=seed).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        final_keras_model.fit(train_full_ds, epochs=optimal_epochs, verbose=1)
        final_keras_model.save(FINAL_KERAS_PATH)

        # --- Phase 3: XGBoost Training ---
        print("Phase 3: Training XGBoost regressor...")
        # Extract features (remove the head)
        feature_extractor = Model(inputs=final_keras_model.inputs, 
                                  outputs=final_keras_model.get_layer('concat_lig_prot').output)
        
        X_train_features = feature_extractor.predict(
            {'lig_input': X_lig_full, 'prot_input': X_prot_full}, batch_size=BATCH_SIZE)
        X_test_features = feature_extractor.predict(
            {'lig_input': X_lig_test, 'prot_input': X_prot_test}, batch_size=BATCH_SIZE)

        xgb_params = {
            'objective': 'reg:squarederror', 
            'n_estimators': 2000, 
            'learning_rate': 0.01, 
            'max_depth': 7, 
            'subsample': 0.7, 
            'colsample_bytree': 0.7, 
            'random_state': seed, 
            'n_jobs': -1,
            # 'device': 'cuda' # Uncomment if XGBoost GPU support is available
        }
        
        final_xgb = xgb.XGBRegressor(**xgb_params)
        final_xgb.fit(X_train_features, y_full_scaled)
        final_xgb.save_model(FINAL_XGB_PATH)

        # --- Evaluation ---
        y_pred_xgb_scaled = final_xgb.predict(X_test_features)
        y_pred_test = scaler_y.inverse_transform(y_pred_xgb_scaled.reshape(-1, 1)).flatten()

        mae = np.mean(np.abs(y_pred_test - y_test))
        mape = np.mean(np.abs((y_pred_test - y_test) / (np.abs(y_test) + 1e-10))) * 100
        pearson, _ = pearsonr(y_pred_test, y_test)
        c_index = concordance_index(y_test, y_pred_test)

        all_results.append({'seed': seed, 'mae': mae, 'mape': mape, 'pearson': pearson, 'c_index': c_index})
        print(f"Seed {seed} Results -> MAE: {mae:.4f}, Pearson: {pearson:.4f}, C-Index: {c_index:.4f}")

        # --- Plotting ---
        abs_errors = np.abs(y_pred_test - y_test)
        sorted_idx = np.argsort(abs_errors)
        percentiles = [25, 50, 75, 90, 95, 100]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Model Performance (Seed {seed})', fontsize=20, fontweight='bold')
        axes = axes.flatten()

        for i, perc in enumerate(percentiles):
            ax = axes[i]
            n_samples = int(len(y_test) * perc / 100)
            idx_subset = sorted_idx[:n_samples]
            y_p_sub = y_pred_test[idx_subset]
            y_t_sub = y_test[idx_subset]
            
            p_metrics = (
                f"Pearson: {pearsonr(y_p_sub, y_t_sub)[0]:.4f}\n"
                f"MAE: {np.mean(np.abs(y_p_sub - y_t_sub)):.4f}"
            )
            
            ax.scatter(y_t_sub, y_p_sub, alpha=0.4, color='steelblue', s=20)
            ax.plot([y_t_sub.min(), y_t_sub.max()], [y_t_sub.min(), y_t_sub.max()], 'r--', lw=2)
            ax.text(0.95, 0.05, p_metrics, transform=ax.transAxes, ha='right', va='bottom', 
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            ax.set_title(f'Top {perc}% ({n_samples} Samples)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(PLOT_PATH, dpi=300)
        plt.close(fig)

        # Cleanup Run
        del final_keras_model, feature_extractor, final_xgb, X_lig_full, X_prot_full
        gc.collect()
        tf.keras.backend.clear_session()

    # 3. Final Aggregation
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("FINAL AGGREGATED RESULTS (Mean +/- SD over 5 seeds)")
    print("="*60)
    
    # Calculate Mean and Std Dev for all metrics
    stats = results_df.agg(['mean', 'std'])
    
    print(f"MAE:      {stats.loc['mean', 'mae']:.4f} +/- {stats.loc['std', 'mae']:.4f}")
    print(f"MAPE:     {stats.loc['mean', 'mape']:.2f}% +/- {stats.loc['std', 'mape']:.2f}%")
    print(f"Pearson:  {stats.loc['mean', 'pearson']:.4f} +/- {stats.loc['std', 'pearson']:.4f}")
    print(f"C-Index:  {stats.loc['mean', 'c_index']:.4f} +/- {stats.loc['std', 'c_index']:.4f}")
    
    print("-" * 60)
    print("Detailed Run Results:")
    print(results_df.round(4).to_string(index=False))
    print("="*60)
    
    print("\n Pipeline execution complete.")


