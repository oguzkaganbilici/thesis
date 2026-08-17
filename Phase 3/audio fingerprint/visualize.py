import matplotlib.pyplot as plt
import numpy as np

def visualize(best_inliers):
    # best_inliers = np.array(best_inliers)
    """
    for seg in best_inliers:
        plt.scatter(seg[:, 0], seg[:, 1], s=2)   # s=1: noktalar küçük, çünkü on binlerce
    """
    plt.plot(best_inliers)
    plt.xlabel("full maç zamanı (frame)")
    plt.ylabel("önem (0/1)")
    plt.show()
