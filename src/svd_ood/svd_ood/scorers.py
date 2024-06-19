import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from scipy.stats import entropy

class Scorers:
    def __init__(self, scorers, temperature=100):
        """
        Args:
            scorers (list): choose from "mcm_score", "l_mcm_score", "gl_mcm_score",
                "energy_score", "entropy_score", "var_score"
            temperature (float): temperature value for softmax
                100 for CLIP scaled logits; 1 for CLIP unscaled logits
        """
        self.temperature = temperature
        self.scorers = scorers
        self.scores = {scorer: [] for scorer in scorers}
        self.to_np = lambda x: x.data.cpu().numpy()

    def get_scores(self):
        """Get scores for each scorer.
        Returns:
            dict: keys are scorers, values are lists of scores,
                each list contains scores for each image
        """
        return self.scores

    def cal_scores(self, logits_global, logits_local=None):
        """Calculate scores for each scorer.
        Args:
            logits_global (tensor): (image_batch_size, text_batch_size)
                cosine similarity scores for global image and text features of CLIP model;
                or global image features of vision-only model
            logits_local (tensor): (image_batch_size, weight, height, text_batch_size)
                cosine similarity scores for local image and text features of CLIP model;
                or global image features of vision-only model
        """
        probs_global = self.to_np(F.softmax(logits_global / self.temperature, dim=1))
        probs_local = self.to_np(F.softmax(logits_local / self.temperature, dim=-1)) \
              if logits_local is not None else None

        for scorer in self.scorers:
            if scorer == "mcm_score":
                score = self.mcm_score(probs_global)
            elif scorer == "l_mcm_score":
                score = self.l_mcm_score(probs_local)
            elif scorer == "gl_mcm_score":
                score = self.gl_mcm_score(probs_global, probs_local)
            elif scorer == "energy_score":
                score = self.energy_score(logits_global)
            elif scorer == "entropy_score":
                score = self.entropy_score(probs_global)
            elif scorer == "var_score":
                score = self.var_score(probs_global)
            else:
                raise NotImplementedError(f"Scorer {scorer} not implemented")
            self.scores[scorer].extend(score.tolist())

    def mcm_score(self, probs_global):
        """https://arxiv.org/pdf/2211.13445.pdf"""
        mcm_global_score = -np.max(probs_global, axis=1)
        return mcm_global_score

    def l_mcm_score(self, probs_local):
        """https://arxiv.org/pdf/2304.04521.pdf"""
        mcm_local_score = -np.max(probs_local, axis=(1, 2, 3))
        return mcm_local_score

    def gl_mcm_score(self, probs_global, probs_local):
        """https://arxiv.org/pdf/2304.04521.pdf"""
        mcm_global_score = -np.max(probs_global, axis=1)
        mcm_local_score = -np.max(probs_local, axis=(1, 2, 3))
        return mcm_global_score + mcm_local_score

    def energy_score(self, logits_global):
        """https://arxiv.org/pdf/2010.03759.pdf"""
        return self.to_np(
            -self.temperature * torch.logsumexp(logits_global / self.temperature, dim=1))

    def entropy_score(self, probs_global):
        return entropy(probs_global, axis=1)

    def var_score(self, probs_global):
        return -np.var(probs_global, axis=1)


def save_scores(scores, file_path, file_name):
    """Save scores into csv file.
    Args:
        scores (dict): keys are scorers, values are lists of scores
        file_path (str): path to save file
        file_name (str): name of save file
    """
    df = pd.DataFrame(scores)
    df.to_csv(os.path.join(file_path, file_name), index=False)


if __name__ == "__main__":
    scorers = [
        "mcm_score", "l_mcm_score", "gl_mcm_score",
        "energy_score", "entropy_score", "var_score"
    ]
    scorers = Scorers(scorers, temperature=1)
    logits_global = torch.rand(2, 3)          # (image_batch_size, text_batch_size)
    logits_local = torch.rand(2, 14, 14, 3)   # (image_batch_size, weight, height, text_batch_size)
    scorers.cal_scores(logits_global, logits_local)
    scores = scorers.get_scores()
    save_scores(scores, "./", "scores.csv")
    print(scores)