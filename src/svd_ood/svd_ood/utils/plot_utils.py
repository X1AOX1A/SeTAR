import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_socres(score_path, scorer):
    scores = pd.read_csv(score_path)[scorer].values.tolist()
    return scores

def plot_distribution(id_scores, ood_scores, cutoff, file_path, file_name,
                      text_shift=[-0.00001, 0], dpi=100):
    """Plot ID and OOD scores distribution.
    Args:
        id_scores (List[float]): ID scores
        ood_scores (List[float]): OOD scores
        cutoff (float): Cutoff value
        file_path (str): Path to save the plot
        file_name (str): Name of the plot
        text_shift (List[float]): Shift the text of cutoff, [x, y]
        dpi (int): Image resolution
    """
    sns.set_theme(style="white", palette="muted")
    palette = ['#A8BAE3', '#55AB83']
    data = {"ID Scores":-1 * np.array(id_scores), "OOD Scores": -1 * np.array(ood_scores)}
    ax = sns.displot(data, kind = "kde", palette=palette, fill=True, alpha=0.8,
                     legend=True, aspect=1.25)
    plt.setp(ax._legend.get_texts(), fontsize='12')
    sns.move_legend(ax, "upper right", bbox_to_anchor=(0.75, 0.9))

    plt.axvline(x=cutoff, color='black', linestyle='--', linewidth=1)
    plt.text(cutoff+text_shift[0], plt.gca().get_ylim()[1]+text_shift[1],
             r'$\lambda$', fontsize=12, color='black')
    plt.savefig(os.path.join(file_path, file_name), bbox_inches='tight', dpi=dpi)


if __name__ == "__main__":
    id_score_file = "./results/backbones/clip_base/ID_ImageNet1K/mcm_score/SeTAR/scores_ID_ImageNet1K_test.csv"
    ood_score_file = "./results/backbones/clip_base/ID_ImageNet1K/mcm_score/SeTAR/scores_OOD_Texture_test.csv"
    cutoff = 0.0010932229924947023
    scorer = "mcm_score"
    id_scores = get_socres(id_score_file, scorer)
    ood_scores = get_socres(ood_score_file, scorer)
    plot_distribution(id_scores, ood_scores, cutoff, "./output", "distribution.png")