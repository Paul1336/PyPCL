import matplotlib.pyplot as plt
import os


def plot_accuracy_vs_k(class_count: int, results_list: list, save_dir: str):
    """
    Plots final test accuracy vs k (number of partial labels) after each (C, k) pair.
    Called incrementally — draws whatever results are available so far for this C.

    Args:
        class_count:  Total number of classes C.
        results_list: List of dicts, each with keys 'k', 'cour', 'mcl'.
        save_dir:     Directory to save the PNG (created if absent).
    """
    os.makedirs(save_dir, exist_ok=True)
    k_values  = [r['k']    for r in results_list]
    cour_accs = [r['cour'] for r in results_list]
    mcl_accs  = [r['mcl']  for r in results_list]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_values, cour_accs, 'b-o', label='Cour 2011 (PLL)', linewidth=2, markersize=6)
    ax.plot(k_values, mcl_accs,  'r-s', label='MCL-LOG (CLL)',   linewidth=2, markersize=6)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title(f'Cour 2011 vs MCL-LOG  —  C = {class_count} classes', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(save_dir, f'C{class_count}_accuracy_vs_k.png')
    fig.savefig(save_path, dpi=150)
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