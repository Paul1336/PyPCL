import matplotlib.pyplot as plt
import os


def plot_accuracy_vs_k(
    class_count: int,
    results_list: list,
    save_dir: str,
    alg1_key: str = 'cour',
    alg1_label: str = 'Cour 2011 (PLL)',
    alg1_style: str = 'b-o',
    alg2_key: str = None,
    alg2_label: str = 'MCL-LOG (CLL)',
    alg2_style: str = 'r-s',
    filename: str = None,
):
    """
    Plots final test accuracy vs k for one or two algorithms.
    Called incrementally — draws whatever results are available so far for this C.

    results_list: list of dicts with keys 'k', <alg1_key>, and optionally <alg2_key>.
    alg2_key:     Pass None to plot a single algorithm.
    filename:     PNG filename override (default: C{class_count}_accuracy_vs_k.png).
    """
    os.makedirs(save_dir, exist_ok=True)
    k_values  = [r['k']      for r in results_list]
    alg1_accs = [r[alg1_key] for r in results_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, alg1_accs, alg1_style, label=alg1_label, linewidth=2, markersize=6)

    if alg2_key is not None:
        alg2_accs = [r[alg2_key] for r in results_list if alg2_key in r]
        k2        = [r['k']      for r in results_list if alg2_key in r]
        if alg2_accs:
            ax.plot(k2, alg2_accs, alg2_style, label=alg2_label, linewidth=2, markersize=6)
        title = f'{alg1_label} vs {alg2_label}  —  C = {class_count} classes'
    else:
        title = f'{alg1_label}  —  C = {class_count} classes'

    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = filename or f'C{class_count}_accuracy_vs_k.png'
    fig.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close(fig)

def save_accuracy_plot(accuracies_dict, epochs_range, args, project_root):
    """Saves the model accuracy plot to a file."""
    plt.figure(figsize=(12, 8))

    for model_name, accuracies in accuracies_dict.items():
        if accuracies:
            plt.plot(epochs_range, accuracies, '-', label=f'{model_name} Test Accuracy')

    # Create title with all experiment arguments.
    args_str = ', '.join(f'{k}={v}' for k, v in vars(args).items())
    plt.title(f'Test Accuracy vs. Epochs\n({args_str})', fontsize=10)
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for long title.

    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a filename from the arguments.
    args_filename = '_'.join(f'{k}_{v}' for k, v in vars(args).items()).replace('.', '_')
    filename = f'accuracy_plot_{args_filename}.png'
    
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)
    plt.close() # Close the figure to free memory.
    print(f"Plot updated and saved to {save_path}")