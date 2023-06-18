def supports_proba(clf):
    """Checks if a given classifier supports the 'predict_proba' method, given a single vector x"""
    return hasattr(clf, 'predict_proba')