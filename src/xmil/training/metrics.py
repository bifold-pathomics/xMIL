import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# todo: we should create a PerformanceMetric class. also for the classifiaction to include precision-recall


class RegressionPerformanceMetric:
    def __init__(self, metric_name=None):
        all_metric_names = [
            "mae",
            "medape",
            "rmse",
            "mse",
            "spearmanr",
            "pearsonr",
            "roc_auc",
        ]

        if isinstance(metric_name, str) and metric_name in all_metric_names:
            metric_name = [metric_name]
        elif isinstance(metric_name, str):
            raise ValueError(
                f"if metric_name is a string it must be among {all_metric_names}"
            )

        if metric_name is None:
            self.metric_name = all_metric_names
        elif isinstance(metric_name, list):
            if set(metric_name).issubset(all_metric_names):
                self.metric_name = metric_name
            else:
                raise ValueError(
                    f"a list of metric names must be a subset of {all_metric_names}"
                )
        else:
            raise ValueError("metric_name must be either a list or a string")

    def get_empty_metric_dict(self):
        metrics = dict()
        for m in self.metric_name:
            if m in ["spearmanr", "pearsonr", "roc_auc"]:
                # these metrics should be maximized
                metrics[m] = -np.Inf
            else:
                # metrics which are desired to be minimized
                metrics[m] = np.Inf
        return metrics

    def compute_metrics_regression(self, targets, preds, ref_value):
        metrics = dict()
        for m in self.metric_name:
            try:
                metrics[m] = self.compute_metric_name_regression(
                    m, targets, preds, ref_value
                )
            except:
                metrics[m] = None
        return metrics

    def compute_metric_name_regression(self, metric_name, targets, preds, ref_value):
        if metric_name == "medape":
            return self.calculate_medape(targets, preds)
        elif metric_name == "mse":
            return self.calculate_mse(targets, preds)
        elif metric_name == "rmse":
            return self.calculate_rmse(targets, preds)
        elif metric_name == "mae":
            return self.calculate_mae(targets, preds)
        elif metric_name == "roc_auc":
            return self.compute_auc_roc_regression(targets, preds, ref_value)
        elif metric_name == "spearmanr":
            return self.calculate_spearmanr(targets, preds)
        elif metric_name == "pearsonr":
            return self.calculate_pearsonr(targets, preds)

        raise NotImplementedError(f"{metric_name} not implemented.")

    @staticmethod
    def compute_auc_roc_regression(targets, preds, ref_value):
        targets = targets.cpu().numpy()
        preds = preds.detach().cpu().numpy()

        if ref_value is None:
            return None
        return roc_auc_score(targets >= ref_value, preds)

    @staticmethod
    def calculate_medape(targets, preds):
        """
        Calculate the Median Absolute Percentage Error (MedAPE).

        :param targets: [torch tensor] target values
        :param preds: [torch tensor] predicted values
        :return: MedAPE: The Median Absolute Percentage Error as a percentage.
        """

        targets = targets.cpu().numpy()
        preds = preds.detach().cpu().numpy()

        # Avoid division by zero by replacing zeros in y_true with a small value
        epsilon = 1e-10
        targets_safe = np.where(targets == 0, epsilon, targets)

        # Calculate absolute percentage errors
        ape = np.abs((targets - preds) / targets_safe) * 100

        # Return the median of the absolute percentage errors
        medape = np.median(ape)

        return medape

    @staticmethod
    def calculate_mse(targets, preds):
        targets = targets.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        return np.mean((targets - preds) ** 2)

    @staticmethod
    def calculate_rmse(targets, preds):
        targets = targets.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        return np.sqrt(np.mean((targets - preds) ** 2))

    @staticmethod
    def calculate_mae(targets, preds):
        targets = targets.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        return np.mean(np.abs(targets - preds))

    @staticmethod
    def calculate_spearmanr(targets, preds):
        targets = targets.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        return spearmanr(targets, preds)[0]

    @staticmethod
    def calculate_pearsonr(targets, preds):
        targets = targets.cpu().numpy().ravel()
        preds = preds.detach().cpu().numpy().ravel()
        return pearsonr(targets, preds)[0]
