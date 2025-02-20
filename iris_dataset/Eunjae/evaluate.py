from dataloader import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

class ClassifierUQEvaluator :
    """ Evaluator object for the iris dataset. 
    
    Parameters
    ----------
    data : dataloader object
        Dataset to use.
    val_portion : float
        Portion of the dataset to leave out as the validation set.
    test_portion : float
        Portion of the dataset to leave out as the test set.
        Sum with the val_portion should not be greater than 1.
    random_seed : int
        Seed to set for the train test split.
    """
    def __init__(
        self,
        data=IrisDataset(),
        val_portion=0.1,
        test_portion=0.2,
        random_seed=42
    ) :
        self.data = data
        self.val_portion = val_portion
        self.test_portion = test_portion
        self.random_seed = random_seed

        if self.val_portion > 1 or self.test_portion > 1 or self.val_portion + self.test_portion > 1 :
            raise ValueError("Portion of validation / test data or their sum should not exceed 1.")

        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self._prep_train_val_test()

    def _prep_train_val_test(self):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.data.X, self.data.y, 
            test_size=self.test_portion, 
            stratify=self.data.y,
            random_state=self.random_seed
        )
        if self.val_portion > 0 :
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=self.val_portion/(1-self.test_portion),
                stratify=y_train_val,
                random_state=self.random_seed
            )
        elif self.val_portion == 0 :
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = None, None
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def neg_log_likelihood(y_true, y_proba):
        return np.mean(-1 * np.log(np.array([y_proba[i, y] for i, y in enumerate(y_true)])))
    
    @staticmethod
    def brier_score(y_true, y_proba):
        if y_proba.shape[1] == 2 : # binary classification
            pass
        elif y_proba.shape[1] > 2 : # multiclass classification
            ohe = OneHotEncoder()
            oharray = ohe.fit_transform(y_true.reshape(-1, 1))
            return np.sum(np.sum(np.square(y_proba - oharray), axis=1)) / len(y_true)
        
    @staticmethod
    def expected_calibration_error(y_true, y_proba, bins=np.arange(0, 1.1, 0.2)):
        """ Computes the expected calibration_error.
        
        Parameters
        ----------
        bins : np.ndarray
            Probability bins to evaluate.
        """
        max_proba = np.max(y_proba, axis=1)
        pred_class = np.argmax(y_proba, axis=1)
        binned = np.digitize(max_proba, bins=bins, right=True)
        
        ece = 0
        for bin_num in np.unique(binned) :
            incidents_of_interest = np.where(binned == bin_num)
            acc = np.sum(pred_class[incidents_of_interest] == y_true[incidents_of_interest])
            rep_proba = np.sum(pred_class[incidents_of_interest])
            ece += (acc - rep_proba) / len(y_true)
        return ece
    
    def evaluate(self, *args) :
        """ Computes the metrics for all models that have been provided as args.
        
        Parameters
        ----------
        args : classifiers.
            Models to evaluate.
        
        Returns
        -------
        None
        """
        for model in args :
            model.fit(self.X_train, self.y_train)
            y_proba = model.predict_proba(self.X_test)
            nll = self.neg_log_likelihood(self.y_test, y_proba)
            brier = self.brier_score(self.y_test, y_proba)
            ece = self.expected_calibration_error(self.y_test, y_proba)
            print(nll, brier, ece)


if __name__ == "__main__" :
    iris_uq = ClassifierUQEvaluator()
    # Comparing different kernels
    iris_uq.evaluate(
        GaussianProcessClassifier(1.0*RBF(1.0), random_state=42, n_jobs=-1),
    )