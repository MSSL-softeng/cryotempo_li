"""plot_table.py"""

import argparse
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter


def load_table_from_pickle(filename: str) -> pd.DataFrame:
    """Load the binned table from a Pickle file.

    Args:
        filename (str): The path to the file from which the table will be loaded.

    Returns:
        pd.DataFrame: The binned median absolute elevation difference table.
    """
    return pd.read_pickle(filename)


def plot_binned_table(binned_table: pd.DataFrame, smooth: bool = False, sigma: float = 1.0) -> None:
    """Plot the binned median absolute elevation difference table as a heatmap with optional
    smoothing,
    correctly oriented axis labels, restricted elevation difference range, an over color, and
    a colorbar with an arrow indicating the over color.

    Args:
        binned_table (pd.DataFrame): The binned median absolute elevation difference table.
        smooth (bool): Whether to apply Gaussian smoothing to the table. Defaults to False.
        sigma (float): The standard deviation for the Gaussian kernel.
        Larger values result in more smoothing. Defaults to 1.0.
    """
    # Optionally smooth the binned table
    if smooth:
        smoothed_values = gaussian_filter(binned_table.values, sigma=sigma)
        binned_table = pd.DataFrame(
            smoothed_values, index=binned_table.index, columns=binned_table.columns
        )

    plt.figure(figsize=(10, 8))

    # Define the colormap with an over color
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_over("red")  # Set the over color to red (you can choose another color if you prefer)

    # Create the heatmap with vmax set to 10 and the custom colormap
    norm = mcolors.Normalize(vmin=binned_table.min().min(), vmax=10)
    heatmap = sns.heatmap(
        binned_table,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        cbar=False,  # Disable the default colorbar
        xticklabels=[f"{x:.1f}" for x in binned_table.columns.astype(float)],
        yticklabels=[f"{y:.1f}" for y in binned_table.index.astype(float)],
        vmax=10,  # Set the maximum value for the colormap
    )

    # Create a ScalarMappable object to use with the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for the ScalarMappable, even if not used

    # Create the colorbar with an arrow for the over color
    cbar = plt.colorbar(sm, ax=heatmap.axes, extend="max")
    cbar.set_label("Elevation Difference (m)")

    # Set the title and labels
    plt.title("Binned Median Absolute Elevation Difference")
    plt.xlabel("Roughness (m)")
    plt.ylabel("Slope (degrees)")

    # Reverse the y-axis
    plt.gca().invert_yaxis()

    # Correct the y-ticks to ensure they match the data orientation
    # plt.yticks(
    #     ticks=np.arange(len(binned_table.index)) + 0.5,
    #     labels=[f'{y:.1f}' for y in binned_table.index.astype(float)][::-1]
    # )

    # Display the plot
    plt.show()


def main():
    """main function"""
    # initiate the command line parser
    parser = argparse.ArgumentParser()

    # add each argument

    parser.add_argument(
        "--ant",
        "-a",
        help=("process Antarctic 2D uncertainty"),
        action="store_true",
    )

    parser.add_argument(
        "--grn",
        "-g",
        help=("process Greenland 2D uncertainty"),
        action="store_true",
    )

    parser.add_argument(
        "--smooth",
        "-s",
        help=("apply gaussian smooth=1.0 to plot"),
        action="store_true",
    )

    # read arguments from the command line
    args = parser.parse_args()

    if not args.ant and not args.grn:
        sys.exit("Must have either --grn or --ant")

    if args.ant:
        area = "ant"
    else:
        area = "grn"

    filename = f"/tmp/{area}_2d_uncertainty_table.pickle"

    retrieved_binned_table = load_table_from_pickle(filename)

    plot_binned_table(retrieved_binned_table, smooth=args.smooth)


if __name__ == "__main__":
    main()
