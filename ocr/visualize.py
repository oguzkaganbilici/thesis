import matplotlib.pyplot as plt

def visualize_labels(arr):
    x = range(len(arr)) # 0'dan N'e sanyie index
    plt.fill_between(x, arr)
    plt.xlabel("saniye")
    plt.savefig("labels.png")