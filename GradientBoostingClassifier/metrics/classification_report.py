def classification_report(y_true, y_pred):
    return {
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "support": len(y_true)
    }