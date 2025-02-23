from dataloader import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
import time

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
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_val = std.transform(X_val)
        X_test = std.transform(X_test)
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def neg_log_likelihood(y_true, y_proba):
        """ Negative log likelihood. Evaluates the quality of model uncertainty on a held out set.
        Lower the better; perfect models will get scores of 0."""
        return np.mean(-1 * np.log(1e-6+np.array([y_proba[i, y] for i, y in enumerate(y_true)])))
    
    @staticmethod
    def brier_score(y_true, y_proba):
        """ Measures the accuracy of predicted probabilities.
        Lower the better; perfect models will get scores of 0."""
        if y_proba.shape[1] == 2 : # binary classification
            pass
        elif y_proba.shape[1] > 2 : # multiclass classification
            ohe = OneHotEncoder()
            oharray = ohe.fit_transform(y_true.reshape(-1, 1))
            return np.sum(np.sum(np.square(y_proba - oharray), axis=1)) / len(y_true)
        
    @staticmethod
    def expected_calibration_error(y_true, y_proba, bins=np.arange(0, 1.1, 0.2)):
        """ Computes the expected calibration_error which measures the correspondence between predicted probabilities
        and emperical accuracy.
        Simply put, it is the gap between the accuracy within a bin and its predicted probability.
        
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
            if np.any(incidents_of_interest):
                acc = np.sum(pred_class[incidents_of_interest] == y_true[incidents_of_interest])
                rep_proba = np.sum(max_proba[incidents_of_interest])
                ece += np.abs((acc - rep_proba)) / len(y_true)
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
            # TODO: incorporate validation set for hyperparameter optimization
            tick = time.time()
            model.fit(self.X_train, self.y_train)
            tock = time.time()
            print(f"Took {round(tock-tick, 3)} seconds to train model.")
            if self.val_portion > 0 :
                print("Validation accuracy", accuracy_score(self.y_val, model.predict(self.X_val)))
            y_proba = model.predict_proba(self.X_test)
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            print("Accuracy:", round(acc, 3))
            nll = self.neg_log_likelihood(self.y_test, y_proba)
            brier = self.brier_score(self.y_test, y_proba)
            ece = self.expected_calibration_error(self.y_test, y_proba)
            print(nll, brier, ece)
            print()
            fig, ax = plt.subplots()
            plt.hist(y_proba[:, 1], bins=np.arange(0, 1.1, 0.2), rwidth=0.95)
            plt.show()


if __name__ == "__main__" :
    iris_uq = ClassifierUQEvaluator(data=StudentDataset())
    # Comparing different kernels
    iris_uq.evaluate(
        GaussianProcessClassifier(3.0*RBF(1.0), random_state=42, n_jobs=-1),
        GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid={
                "n_estimators":[10, 25, 50],
                "max_depth":[2, 3, 5, None]
            },
            n_jobs=-1
        ),
        GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid = {
                "C":[0.1, 0.3, 1, 3, 10],
                "kernel":["rbf", "sigmoid"]
            }
        )
    )