import os
import json
import pandas as pd

class NestedDict:
    """A nested dictionary class.
    Example:
    nested_dict = NestedDict()
    nested_dict['key1']['key2']['key3'] = [1]

    normal_dict = nested_dict()  # convert to normal dict
    print(normal_dict)           # {'key1': {'key2': {'key3': [1]}}}
    """
    def __init__(self):
        self._data = {}

    def __getitem__(self, key):
        if key not in self._data:
            self._data[key] = NestedDict()
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def to_dict(self):
        result = {}
        for key, value in self._data.items():
            if isinstance(value, NestedDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __call__(self):
        return self.to_dict()


class MetricHelper:
    """Helper class for metrics."""
    def __init__(self):
        super().__init__()
        # below should be init be init_best_metric
        self.factor = None
        self.best_metrics = {}
        self.best_metric_name = None

    def sanity_check(self, best_metric_name):
        """Check if the best_metric_name is valid."""
        support_metrics = ["locoop_loss", "loss_id", "loss_ood", "auroc", "aupr", "fpr"]
        assert best_metric_name in support_metrics, \
            f"best_metric_name {best_metric_name} is not in supported metrics {support_metrics}."

    def init_best_metric(self, best_metric_name):
        """Initialize the best_metric and factor.
        Args:
            best_metric_name (str): best metric to select the best settings
        """
        self.sanity_check(best_metric_name)
        init_values = {
            "locoop_loss": float('inf'),  # locoop loss
            "loss_id": float('inf'),      # id loss
            "loss_ood": float('inf'),     # ood loss
            "auroc": 0,                   # area under the ROC curve,
            "aupr":  0,                   # area under the precision-recall curve
            "fpr":   100,                 # false positive rate
        }
        factors = {
            "locoop_loss": -1,  # lower is better
            "loss_id": -1,      # lower is better
            "loss_ood": -1,     # lower is better
            "auroc": 1,         # higher is better
            "aupr":  1,         # higher is better
            "fpr":   -1,        # lower is better
        }
        self.best_metric_name = best_metric_name
        self.best_metrics = init_values
        self.factor = factors[best_metric_name]

    def is_better(self, current_metrics):
        """Check if the metrics is better than the current best_metric.
        Args:
            current_metrics (dict): current metric values, e.g.:
                {aurouc, aupr, fpr, cutoff} or {locoop_loss, loss_id, loss_ood}
        Returns:
            bool: True if the metrics is better than the current best_metric, otherwise False.
        """
        # if the current metric is better, return True
        if current_metrics[self.best_metric_name]*self.factor > \
           self.best_metrics[self.best_metric_name]*self.factor:
            return True
        else:
            return False

    def update_best_metrics(self, best_metrics):
        """Update the best_metrics.
        Args:
            dict: best metric value, e.g.:
                {aurouc, aupr, fpr, cutoff} or {locoop_loss, loss_id, loss_ood}
        """
        self.best_metrics = best_metrics

    def get_best_metrics(self, add_star=True):
        """Get the best_metrics.
        Args:
            add_star (bool): add '*' to the best_metrics keys if True
        Returns:
            dict: best metric value, e.g.:
                {aurouc, aupr, fpr, cutoff} or {locoop_loss, loss_id, loss_ood}
        """
        if add_star:
            return {f"{k}*" if k == self.best_metric_name else k: v \
                    for k, v in self.best_metrics.items()}
        else:
            return self.best_metrics


class SettingHelper:
    """Helper class for lora setting in search process."""
    def __init__(self):
        super().__init__()
        self.settings = []      # store the best lora settings

    def add_setting(self, tower_type, weight_type, layer_num, reduc_ratio):
        self.settings.append([tower_type, weight_type, layer_num, reduc_ratio])

    def pop_setting(self):
        self.settings.pop()

    def get_settings(self):
        """Get lora settings.
        Returns:
            list: [[tower_type, weight_type, layer_num, reduc_ratio], ...]
        """
        return self.settings


class ResultHelper:
    """Helper class for results in search process."""
    def __init__(self):
        super().__init__()
        self.log = NestedDict() # store the settings and metrics that computed
        self.best_log = {}      # store the best setting and metrics of each step

    def add_result(self, step, metrics_dicts, tower_type, weight_type, layer_num, reduc_ratio):
        """Add the setting and metrics that computed.
        Args:
            step (int): the determined step
            metrics_dicts (dict): metrics for the given low rank settings, e.g.:
                {"loss": {locoop_loss, loss_id, loss_ood}, "scorers": {aurouc, aupr, fpr, cutoff}}
            tower_type (str): "visual" or "text"
            weight_type (str): "W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"
            layer_num (int): layer number
            reduc_ratio (float): percentage of large singular values to drop
        """
        for scorer, metrics in metrics_dicts.items():
            if type(layer_num) is tuple:    # (layer_num, block_num) for SwinTransformer
                layer_num, block_num = layer_num
                self.log[step][tower_type][weight_type][layer_num][block_num][reduc_ratio][scorer] = metrics
            else:
                self.log[step][tower_type][weight_type][layer_num][reduc_ratio][scorer] = metrics

    def save_results(self, file_path, file_name):
        """Save the setting and metrics that computed to a json file."""
        dict_ = self.log()
        with open(os.path.join(file_path, file_name), 'w') as f:
            json.dump(dict_, f, indent=2)

    def add_best_results(self, step, best_metrics, tower_type, weight_type, layer_num, best_ratio):
        """Add the best setting and metrics of each step.
        Args:
            step (int): step number
            best_metrics (dict): best metric values, e.g.:
                {aurouc, aupr, fpr, cutoff} or {locoop_loss, loss_id, loss_ood}
            tower_type (str): "visual" or "text"
            weight_type (str): "W_q", "W_k", "W_v", "W_o", "W_up", "W_down", "W_p"
            layer_num (int): layer number
            best_ratio (float): percentage of large singular values to drop
        """
        self.best_log[step] = {
            "tower_type": tower_type,
            "weight_type": weight_type,
            "layer_num": layer_num,
            "best_ratio": best_ratio,
        }
        self.best_log[step].update(best_metrics)

    def get_best_results(self):
        """Get step log with metrics and best setting of each step.
        Returns:
            dict: {step: {"tower_type", "weight_type", "layer_num", "best_ratio",
                          "aurouc", "aupr", "fpr"}}
        """
        return self.best_log

    def get_best_results_string(self):
        """Get step log with metrics and best setting of each step in string format."""
        df = pd.DataFrame.from_dict(self.best_log, orient='index')
        df.index.name = "step"
        return df.to_string()

    def save_best_results(self, file_path, file_name):
        """Save the best setting and metrics of each step to a csv file."""
        df = pd.DataFrame.from_dict(self.best_log, orient='index')
        df.index.name = "step"
        df.to_csv(os.path.join(file_path, file_name), index=True)