import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class confusion_matrix_display:

    def __init__(self, confusion_matrix, display_labels=None):
        self.confusion_matrix  = confusion_matrix
        self.display_labels    = display_labels

    def plot(self, title="Confusion Matrix", save_path=None):
        cm     = self.confusion_matrix
        labels = self.display_labels if self.display_labels is not None \
                 else [str(i) for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        thresh = cm.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color=color, fontsize=13)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  ✓ Đã lưu: {save_path}")
        plt.close()
