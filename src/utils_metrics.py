from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import numpy as np

def compute_metrics(y_true, y_scores):
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5
    
    # Find best F1 by searching thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    
    return {
        'auc': auc,
        'f1': best_f1,
        'best_threshold': best_threshold
    }
