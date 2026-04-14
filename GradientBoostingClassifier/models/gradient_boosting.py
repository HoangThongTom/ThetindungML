class GradientBoostingClassifier:
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        self.models = []
        self.init_pred = None

    def fit(self, X, y):
        self.init_pred = 0
        
        y_pred = [self.init_pred] * len(y)
        # boosting loop
        for i in range(self.n_estimators):
            
            residuals = None 

            model = None

            pred = None
            # Update predictions theo bossting
            #....
            self.models.append(model)

    def predict(self, X):
        y_pred = [self.init_pred] * len(X)

        for model in self.models:
            pred = None
            # Cập nhật y_pred theo boosting
            #....

        # nhớ chuyển về nhãn (classification)
        # ....
        return [0] * len(X)