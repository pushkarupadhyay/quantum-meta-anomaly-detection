from sklearn.linear_model import LogisticRegression
import numpy as np

class FewShotHead:
    def __init__(self, class_weight='balanced'):
        self.model = LogisticRegression(class_weight=class_weight, max_iter=1000)

    def fit(self, X_support, y_support):
        # Handle cases where there might be only one class in support set (though meta-learning sampling should prevent this)
        if len(np.unique(y_support)) < 2:
            # Fallback if only one class is present
            self.model = None
            self.constant_prediction = y_support[0]
        else:
            self.model.fit(X_support, y_support)
            self.model = self.model # Ensure it's not None

    def predict_proba(self, X_query):
        if hasattr(self, 'model') and self.model is not None:
            # Check if model was actually fitted
            try:
                return self.model.predict_proba(X_query)[:, 1]
            except:
                # Fallback if model not fitted
                return np.zeros(len(X_query))
        else:
            # Return constant prediction if model wasn't fitted
            return np.full(len(X_query), self.constant_prediction)

    def predict(self, X_query, threshold=0.5):
        probas = self.predict_proba(X_query)
        return (probas >= threshold).astype(int)
