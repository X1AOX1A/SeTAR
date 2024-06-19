import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
import sklearn.metrics as sk


class Metrics:
    def __init__(self, recall_level=0.95):
        """Metrics class to evaluate OOD detection performance.
        Args:
            recall_level (float): recall level to compute FPR
        """
        self.recall_level = recall_level

    def compute_metrics(self, id_scores, ood_scores):
        """Evaluate OOD detection performance (FPR95, AUROC, AUPR).
        Args:
            id_scores (list): ID dataset scores
            ood_scores (list): OOD dataset scores
        Returns:
            metrics (dict):
                auroc (float): area under the ROC curve
                aupr (float): area under the precision-recall curve
                fpr (float): false positive rate at recall level
                cutoff (float): threshold cutoff at recall level
        """
        logging.debug(f"ID scores: {stats.describe(id_scores)}")
        logging.debug(f"OOD scores: {stats.describe(ood_scores)}")
        pos = -np.array(id_scores).reshape((-1, 1))
        neg = -np.array(ood_scores).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        aurocs = sk.roc_auc_score(labels, examples)
        auprs = sk.average_precision_score(labels, examples)
        fprs, cutoff = self.fpr_and_fdr_at_recall(labels, examples, self.recall_level)
        auroc, aupr, fpr = np.mean(aurocs), np.mean(auprs), np.mean(fprs)
        return {
            "auroc": auroc*100,
            "aupr": aupr*100,
            "fpr": fpr*100,
            "cutoff": cutoff
        }

    def print_metrics(self, metrics, name):
        """Print metrics to the logging.
        Args:
            metrics (dict): metrics, {"auroc": float, "aupr": float, "fpr": float}
            name (str): logging info
        """
        auroc, aupr, fpr = metrics["auroc"], metrics["aupr"], metrics["fpr"]
        logging.info(name)
        logging.info('  FPR{:d} AUROC AUPR'.format(int(100 * self.recall_level)))
        logging.info('& {:.2f} & {:.2f} & {:.2f}'.format(fpr, auroc, aupr))

    def save_metrics(self, metrics_dict, file_dir, file_name):
        """Save and print metrics to a csv file.
        Args:
            metrics_dict: metrics_dict[ood_dataset, Avg] =
                          {"auroc": float, "aupr": float, "fpr": float}
        """
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        # delete "cutoff" column
        df = df.drop(columns=["cutoff"], errors='ignore')
        # rename columns
        columns = {"auroc": "AUROC", "aupr": "AUPR",
                   "fpr": "FPR{:d}".format(int(100 * self.recall_level))}
        df = df.rename(columns=columns)
        df.index.name = "OOD dataset"
        # reorder columns
        df = df[["FPR{:d}".format(int(100 * self.recall_level)), "AUROC", "AUPR"]]
        df.to_csv(os.path.join(file_dir, file_name), float_format='%.2f')
        logging.info("############ Mean metrics ############")
        logging.info("\n"+df.to_string(index=True, float_format='%.2f'))
        logging.info(f"Metrics saved to {os.path.join(file_dir, file_name)}")

    def save_cutoffs(self, cutoff_dict, file_dir, file_name):
        """Save and print cutoffs to a csv file.
        Args:
            cutoff_dict: cutoff_dict[ood_dataset] = threshold
        """
        df = pd.DataFrame(cutoff_dict.items(), columns=["OOD dataset", "Cutoff"])
        df.to_csv(os.path.join(file_dir, file_name), index=False)
        logging.info("############ Thresholds Cut-off ############")
        logging.info("\n"+df.to_string(index=False))
        logging.info(f"Thresholds cutoff saved to {os.path.join(file_dir, file_name)}")

    def fpr_and_fdr_at_recall(self, y_true, y_score, recall_level=0.95, pos_label=None):
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                    np.array_equal(classes, [-1, 1]) or
                    np.array_equal(classes, [0]) or
                    np.array_equal(classes, [-1]) or
                    np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = self.stable_cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

        thresholds = y_score[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)  # [last_ind::-1]
        recall, fps, tps, thresholds = \
            np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

        cutoff = np.argmin(np.abs(recall - recall_level))
        fpr_at_recall = fps[cutoff] / (np.sum(np.logical_not(y_true)))
        cutoff = thresholds[cutoff]  # thresholds cutoff when TPR(recall)=0.95
        return fpr_at_recall, cutoff

    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)
        expected = np.sum(arr, dtype=np.float64)
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                            'its last element does not correspond to sum')
        return out


if __name__ == "__main__":
    from svd_ood.utils.logger import setup_logger
    setup_logger(log_file=None)
    metrics = Metrics(recall_level=0.95)
    id_scores = np.random.rand(100)
    ood_datasets = ["OOD_iNaturalist", "OOD_Sun", "OOD_Places", "OOD_Texture"]
    best_score = "mcm_score"
    metrics_dict, cutoff_dict = {}, {}
    for ood_dataset in ood_datasets:
        ood_scores = np.random.rand(100)
        metrics_ = metrics.compute_metrics(id_scores, ood_scores)
        metrics.print_metrics(metrics_, f"{ood_dataset} - {best_score}")
        metrics_dict[ood_dataset] = metrics_
        cutoff_dict[ood_dataset] = metrics_["cutoff"]
    metrics.save_metrics(metrics_dict, "./", f"metrics_{best_score}.csv")
    metrics.save_cutoffs(cutoff_dict, "./", f"cutoffs_{best_score}.csv")