import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from loguru import logger

def evaluate_peak_detection(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    predicted_probabilities: Optional[pd.Series] = None, # Optional: for AUC, PR-AUC
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Evaluates the performance of a peak detection model.
    
    Args:
        true_labels (pd.Series): The ground truth binary labels (1 for peak, 0 for not peak).
        predicted_labels (pd.Series): The binary predictions from the model.
        predicted_probabilities (pd.Series, optional): Probabilities of the positive class.
                                                       Required for ROC AUC and PR AUC.
        prefix (str): Prefix for metric names (e.g., 'test_').

    Returns:
        Dict[str, Any]: A dictionary of evaluation metrics.

    Note:
        Labels and probabilities are reindexed to align with `true_labels`'s index.
        Data points in `true_labels` not found in predictions will be treated as negatives (0/0.0).
        Data points in predictions not found in `true_labels` will be ignored.
    """
    if not true_labels.index.equals(predicted_labels.index):
        logger.warning("Indices of true_labels and predicted_labels do not match. Attempting reindex.")
        predicted_labels = predicted_labels.reindex(true_labels.index).fillna(0).astype(int)
        if predicted_probabilities is not None:
            predicted_probabilities = predicted_probabilities.reindex(true_labels.index).fillna(0.0)

    metrics = {}

    try:
        # Core classification metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(true_labels, predicted_labels)
        metrics[f'{prefix}precision'] = precision_score(true_labels, predicted_labels, zero_division=0)
        metrics[f'{prefix}recall'] = recall_score(true_labels, predicted_labels, zero_division=0)
        metrics[f'{prefix}f1_score'] = f1_score(true_labels, predicted_labels, zero_division=0)
        
        # Confusion matrix components
        # Handle cases where confusion_matrix might return differently (e.g., single class)
        cm = confusion_matrix(true_labels, predicted_labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1): # Only one class present
            if true_labels.iloc[0] == 0: # Only negatives
                tn, fp, fn, tp = cm[0,0], 0, 0, 0
            else: # Only positives
                tn, fp, fn, tp = 0, 0, 0, cm[0,0]
        else: # Should not happen with binary classification, but as fallback
            tn, fp, fn, tp = 0, 0, 0, 0
            logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")

        # Removed redundant int() casts as confusion_matrix already returns integers
        metrics[f'{prefix}true_positives'] = tp
        metrics[f'{prefix}false_positives'] = fp
        metrics[f'{prefix}true_negatives'] = tn
        metrics[f'{prefix}false_negatives'] = fn
        
        # Specificity / True Negative Rate
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Optional: AUC ROC and PR AUC if probabilities are provided
        if predicted_probabilities is not None and not predicted_probabilities.empty:
            # Check if true_labels contains at least two classes for AUC calculation
            if len(true_labels.unique()) > 1:
                try:
                    metrics[f'{prefix}roc_auc'] = roc_auc_score(true_labels, predicted_probabilities)
                except ValueError as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}. (Likely only one class present in true labels after filtering)")
                    metrics[f'{prefix}roc_auc'] = float('nan')
                
                try:
                    metrics[f'{prefix}pr_auc'] = average_precision_score(true_labels, predicted_probabilities)
                except ValueError as e:
                    logger.warning(f"Could not calculate PR AUC: {e}. (Likely only one class present in true labels after filtering)")
                    metrics[f'{prefix}pr_auc'] = float('nan')
            else:
                logger.warning("ROC AUC and PR AUC cannot be calculated: Only one class present in true labels.")
                metrics[f'{prefix}roc_auc'] = float('nan')
                metrics[f'{prefix}pr_auc'] = float('nan')
        else:
            metrics[f'{prefix}roc_auc'] = float('nan')
            metrics[f'{prefix}pr_auc'] = float('nan')
            logger.info("Predicted probabilities not provided or empty. Skipping ROC AUC and PR AUC.")
        
    except Exception as e:
        logger.error(f"Error calculating peak detection metrics: {e}", exc_info=True)
        # Populate with NaN or default values if an error occurs
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'true_positives', 'false_positives', 
                    'true_negatives', 'false_negatives', 'specificity', 'roc_auc', 'pr_auc']:
            metrics[f'{prefix}{key}'] = float('nan')

    logger.info("Peak detection evaluation completed.")
    return metrics