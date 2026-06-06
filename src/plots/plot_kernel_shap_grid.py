import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_kernel_shap_grid(root_dir, output_path=None):
    root_dir = Path(root_dir)
    segment_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir()],
        key=lambda d: int(''.join(filter(str.isdigit, d.name)) or 0)
    )
    baseline_order = ["zeros", "blurred", "average", "ones"]

    n_rows = len(segment_dirs)
    n_cols = len(baseline_order)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"No folders found in {root_dir}")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, segment_dir in enumerate(segment_dirs):
        segment_label = segment_dir.name
        for col_idx, baseline_name in enumerate(baseline_order):
            ax = axes[row_idx][col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            image_dir = segment_dir / baseline_name
            if image_dir.exists() and image_dir.is_dir():
                image_files = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
                if image_files:
                    img = mpimg.imread(image_files[0])
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'No image', ha='center', va='center', color='red', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'Missing folder', ha='center', va='center', color='red', fontsize=12)

            if row_idx == 0:
                ax.set_title(baseline_name, pad=12)
            if col_idx == 0:
                ax.set_ylabel(segment_label, rotation=90, labelpad=40, va='center')

    plt.tight_layout()
    if output_path is None:
        output_path = root_dir / 'kernel_shap_grid.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot Kernel SHAP results in a grid.')
    parser.add_argument('--root', type=str, default='xai_results/cat3/kernel_shap',
                        help='Root folder containing segment and baseline subfolders.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for the saved grid image.')
    args = parser.parse_args()

    root_path = Path(args.root)
    saved_path = plot_kernel_shap_grid(root_path, output_path=args.output)
    print(f'Grid saved to: {saved_path}')
