import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from moabb.datasets import BNCI2014_004, BNCI2015_001
from moabb.paradigms import MotorImagery
from torch.utils.data import DataLoader, TensorDataset
from pyriemann.estimation import Covariances
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from braindecode.preprocessing import exponential_moving_standardize

from utils.spd_utils import verify_spd_properties
from utils.utils import set_seed, get_device
from data.riemannian_parallel_transport import RiemannianParallelTransport
from visualization.visualizer import EegVisualizer

class MicrovoltsScaler(BaseEstimator, TransformerMixin):
    """Converts volts to microvolts (standard EEG unit)"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.astype(float)
        return X * 1e6  # multiply by 1e6 to convert V to μV


class EMSTransformer(BaseEstimator, TransformerMixin):
    """Exponential Moving Standardization"""
    def __init__(self, factor_new=0.01, init_block_size=1000, epsilon=1e-6):
        self.factor_new = factor_new
        self.init_block_size = init_block_size
        self.epsilon = epsilon
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([
            exponential_moving_standardize(
                trial.T.astype(np.float64),  # requires double precision
                factor_new=self.factor_new,  # controls adaptation rate
                init_block_size=self.init_block_size,  # initial normalization window
                eps=self.epsilon  # prevents division by zero
            ).T
            for trial in X
        ])


def load_mi_dataset(dataset_name, subjects=None, n_classes=2, fmin=8, fmax=30, tmin=3.0, tmax=5.0):
    """Load Motor Imagery dataset"""

    if dataset_name == "BNCI2014_004":
        dataset = BNCI2014_004()
    elif dataset_name == "BNCI2015_001":
        dataset = BNCI2015_001()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Using dataset: {dataset_name} with {len(dataset.subject_list)} subjects")
    
    # use all subjects if none are specified
    selected_subjects = subjects if subjects is not None and len(subjects) > 0 else dataset.subject_list
    print(f"Loading data for subjects: {selected_subjects}")

    paradigm = MotorImagery(
        n_classes=n_classes,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        events=dataset.event_id,
        resample=None # keep original sampling rate  
    )

    print(f"Paradigm: Freq={fmin}-{fmax}Hz, Time={tmin}-{tmax}s, Classes={n_classes}")
    print(f"Dataset event_id: {dataset.event_id}")
    
    try:
        X_raw, y_raw_labels, meta = paradigm.get_data(
            dataset=dataset,
            subjects=selected_subjects,
            return_epochs=False # returns numpy array
        )
        print(f"Successfully loaded {len(X_raw)} raw epochs/trials")
        return X_raw, y_raw_labels, meta
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocessing_pipeline(config, exclude_subject=None, visualize_first_trial=False):
    """
    Loads, splits, preprocesses, aligns (optional), and creates dataloaders.
    Handles both LOSO and train/test split!

    Args:
        config: The main configuration object.
        exclude_subject (int, optional): Subject ID to exclude for LOSO.
        visualize_first_trial (bool): If True, generates a plot of the preprocessing steps.

    Returns:
        dict: A dictionery containing all relevant data stages and loaders.
    """

    set_seed(config.data.seed)
    device = get_device(config.device)
    print(f"Using device: {device}")
    

    # --- load raw data ---
    print("\n--- Loading raw data via MOABB ---")
    X_raw, y_raw_labels, meta = load_mi_dataset(
        dataset_name=config.data.dataset,
        subjects=config.data.subjects,
        n_classes=config.data.n_classes,
        fmin=config.data.fmin,
        fmax=config.data.fmax,
        tmin=config.data.tmin,
        tmax=config.data.tmax
    )

    # --- extract subject metadata ---
    if 'subject' not in meta:
        raise ValueError("Subject metadata key 'subject' not found in loaded metadata.")

    subjects = meta['subject'].values
    unique_subject_ids = np.unique(subjects)
    print(f"Subject metadata extracted: {len(unique_subject_ids)} unique subjects found.")

    # --- define label map ---
    # uses numeric mapping {0, 1} for standard VAE/classifier input
    if config.data.dataset == "BNCI2014_004":
        label_map = {'right_hand': 0, 'left_hand': 1}
    elif config.data.dataset == "BNCI2015_001":
        label_map = {'right_hand': 0, 'feet': 1}
    else:
        raise ValueError(f"Label mapping not defined for dataset: {config.data.dataset}")
    print(f"Using label mapping: {label_map}")

    # --- split raw data ---
    is_loso = exclude_subject is not None
    if is_loso:
        print(f"\n--- Performing LOSO Split: Excluding Subject {exclude_subject} ---")
        if exclude_subject not in unique_subject_ids:
            raise ValueError(f"Subject {exclude_subject} to exclude not found in loaded subjects: {unique_subject_ids}")

        test_mask = (subjects == exclude_subject)
        train_mask = ~test_mask

        if not np.any(test_mask):
            raise ValueError(f"No data found for test subject {exclude_subject}.")
        if not np.any(train_mask):
            raise ValueError(f"No training data found after excluding subject {exclude_subject}.")

        X_train_raw, y_train_raw_labels, subjects_train = X_raw[train_mask], y_raw_labels[train_mask], subjects[train_mask]
        X_test_raw, y_test_raw_labels, subjects_test = X_raw[test_mask], y_raw_labels[test_mask], subjects[test_mask]
    else:
        print("\n--- Performing Train/Test Split ---")
        # stratify ensures class proportions equal in train/test splits
        X_train_raw, X_test_raw, y_train_raw_labels, y_test_raw_labels, subjects_train, subjects_test = train_test_split(
            X_raw, y_raw_labels, subjects,
            test_size=config.data.test_size,
            random_state=config.data.seed,
            stratify=y_raw_labels
        )
    print(f"Raw data split result: Train={len(X_train_raw)}, Test={len(X_test_raw)}")

    # --- define and apply preprocessing ---
    print("\n--- Preprocessing Data (Scaling -> EMS -> Covariance Estimation) ---")
    
    scaler = MicrovoltsScaler()
    standardizer = EMSTransformer()
    cov_estimator = Covariances(estimator='oas')

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_train_standardized = standardizer.fit_transform(X_train_scaled)
    cov_estimator.fit(X_train_standardized, y_train_raw_labels) # Fit on training data
    X_train_cov = cov_estimator.transform(X_train_standardized)
    X_test_cov = cov_estimator.transform(standardizer.transform(scaler.transform(X_test_raw)))

    ''' UNCOMMENT FOR VISUALIZATION PLOTS !!!
    if visualize_first_trial and len(X_train_raw) > 0 and (exclude_subject is None or exclude_subject == 2):
        print("\n--- Generating Preprocessing Visualization for First Training Trial ---")
        visualizer = EegVisualizer(output_dir=config.paths._visualizations_dir)
        sampling_rate = X_train_raw.shape[2] / (config.data.tmax - config.data.tmin)
        
        if exclude_subject is None:
            trial_subject = subjects_train[0]
            trial_id_str = f"Subject {trial_subject} Within-Subject Study"
        else:
            trial_subject = subjects_train[1]
            trial_id_str = f"Subject {trial_subject} From LOSO Fold 2 Study"

        visualizer.plot_eeg_transformations(
            raw_epoch_data=X_train_scaled[0],
            standardized_epoch_data=X_train_standardized[0],
            cov_matrix=X_train_cov[0],
            sampling_rate=sampling_rate,
            trial_id=trial_id_str,
            name_prefix=f"{config.name}_preprocessing_showcase"
        )
    '''
    
    # keep copies for the return dictionary
    X_train_processed = X_train_cov.copy()
    X_test_processed = X_test_cov.copy()

    print("\n--- Verifying Train SPD properties ---")
    verify_spd_properties(X_train_processed, verbose=True)
    print("\n--- Verifying Test SPD properties ---")
    verify_spd_properties(X_test_processed, verbose=True)

    # --- convert labels to numeric label map ---
    try:
        y_train = np.array([label_map[label] for label in y_train_raw_labels])
        y_test = np.array([label_map[label] for label in y_test_raw_labels])
    except KeyError as e:
        print(f"Error: Label '{e}' not found in label_map {label_map}. Check dataset labels.")
        raise

    # --- apply alignment ---
    transport_object = None
    alignment_enabled = hasattr(config, 'alignment') and config.alignment.enabled
    if alignment_enabled:
        print("\n--- Apply Riemannian parallel transport ---")
        transport_object = RiemannianParallelTransport(epsilon=config.alignment.epsilon)

        # fit alignment ONLY on training covariance matrices
        print(f"Fitting transport on {len(X_train_processed)} training samples...")
        # exclude_subject=None because X_train is already the training set
        transport_object.fit(X_train_processed, y_train, subjects_train, exclude_subject=None)

        # transform both train and test covariance matrices
        print("Transforming training and test data...")
        X_train_processed = transport_object.transform(X_train_processed, subjects_train)
        X_test_processed = transport_object.transform(X_test_processed, subjects_test)

        print("Verifying SPD properties after transport...")
        verify_spd_properties(X_train_processed, verbose=True)
        verify_spd_properties(X_test_processed, verbose=True)
        print("Transport applied.")
    else:
        print("\nSkipping Riemannian parallel transport.")

    # -- create dataloaders ---
    print("\n--- Creating dataloaders ---")
    try:
        X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    except TypeError as e:
        print(f"Error converting data to tensors: {e}")
        print("Check data types after preprocessing and alignment.")
        raise

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    generator = torch.Generator().manual_seed(config.data.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, 
                    shuffle=True, generator=generator, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size,
                    shuffle=False, drop_last=False)

    print(f"Dataloaders created: Train batches={len(train_loader)}, Test batches={len(test_loader)}.")

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "X_train_cov_unaligned": X_train_cov,
        "X_test_cov_unaligned": X_test_cov,
        "X_train": X_train_processed,  # final data for training (aligned or unaligned)
        "y_train": y_train,
        "subjects_train": subjects_train,
        "X_test": X_test_processed,    # final data for testing (aligned or unaligned)
        "y_test": y_test,
        "subjects_test": subjects_test,
        "aligner": transport_object
    }
