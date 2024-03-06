import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from utils.util_data import load_dataset


def get_cmap(num_colors):
    if num_colors <= 10:
        cm_name = "tab10"
    elif num_colors <= 20:
        cm_name = "tab20"
    else:
        assert False
    return cm.get_cmap(cm_name)

def analyze_dataset(dataset_path, output_dir):
    dataset = load_dataset(dataset_path)
    
    #-----------------------------
    # Stepwise frequency analysis 
    #-----------------------------
    max_steps = len(dataset[0][0]) # num_nodes
    num_labels = 2
    freq = [[] for _ in range(num_labels)]
    weights = [[] for _ in range(num_labels)]
    for instance in dataset:
        labels = instance[-1]
        for step, label in labels:
            freq[label].append(step)
    # visualize histogram
    fig = plt.figure(figsize=(10, 10))
    binwidth = 1
    bins = np.arange(0, max_steps + binwidth, binwidth)
    cmap = get_cmap(num_labels)
    for i in range(len(weights)):
        weights[i] = np.ones(len(freq[i])) / len(dataset)
        plt.hist(freq[i], bins=bins, alpha=0.5, weights=weights[i], ec=cmap(i), color=cmap(i), label="prioritizing tour length", align="left")
    plt.xlabel("Steps")
    plt.ylabel("Frequency (density)")
    if max_steps <= 20:
        plt.xticks(np.arange(0, max_steps+1, 1))
    plt.title(f"# of samples = {len(dataset)}\n# of nodes = {max_steps}")
    plt.legend()
    plt.savefig(f"{output_dir}/hist.png", dpi=150, bbox_inches="tight")

    #-----------------------------
    # Overall ratio of each class
    #-----------------------------
    total = np.sum([len(freq[i]) for i in range(num_labels)])
    ratio = np.array([len(freq[i]) for i in range(num_labels)])
    ratio = ratio / total
    with open(f"{output_dir}/ratio.dat", "w") as f:
        for i in range(len(ratio)):
            print(f"label{i}, {ratio[i]}", file=f)

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        dataset_dir = os.path.split(args.dataset_path)[0]
        output_dir = dataset_dir
    else:
        output_dir = args.output_dir
    output_dir += "/analysis"
    os.makedirs(output_dir, exist_ok=True)
    analyze_dataset(args.dataset_path, output_dir)