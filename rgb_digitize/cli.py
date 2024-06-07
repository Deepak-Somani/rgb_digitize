import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import argparse
import os

from rgb_digitze.rgb_tools import (
    extract_rgb,
    interpolate_data,
    plot_rgb,
)


def extract_scale_subparser_func(args):
    """Driver function that provides you with a reference scale based on scale image provided

    Parameters
    ----------
    args.image_path: str
        Scale Image Path
    args.upper_limit: float
        Upper Limit for scale
    args.lower_limit: float
        Lower Limit for scale
    args.output_file: str
        Dataframe file to store (r,g,b) vs data values
    """

    print(
        "I am extracting the values of the scale: {0}, based on limits provided as [{1},{2}]".format(
            args.image_path, args.upper_limit, args.lower_limit
        )
    )

    rgb_scale = extract_rgb(img=Image.open(args.image_path), color_scale=True)
    rgb_scale_df = pd.DataFrame(rgb_scale, columns=["r", "g", "b", "x", "y"]).drop(
        columns=["x", "y"]
    )
    rgb_scale_df["data"] = np.linspace(
        args.upper_limit, args.lower_limit, num=rgb_scale_df.shape[0]
    )
    rgb_scale_df = rgb_scale_df.drop_duplicates(
        subset=["r", "g", "b"], ignore_index=True
    )

    rgb_scale_df.to_csv(args.output_file)

    print("Done!")


def extract_indices_subparser_func(args):
    """Driver function to extract the data values of index images

    Parameters
    ----------
    args.scale_file: str
        Scale DataFrame file
    args.y_start: int
        Start of Y-index for the images
    args.x_start: int
        Start of X-index for the images
    args.y_end: int
        End of Y-index for the images
    args.x_end: int
        End of X-index for the images
    args.image_folder: str
        Index Images Folder
    args.output_file: str
        Dataframe file to store data values vs (x_index, y_index)

    NOTES
    -----
    The index images are hard-coded in this version of code as: (x-index)_(y-index).png
    """

    rgb_scale_df = pd.read_csv(args.scale_file)

    # list of dictionaries to store the data-values corresponding to x-index and y-index
    dist_store = []

    # Different cells of the colormap should be provided in the form of "(y-index)_(x-index).png"
    y_index_start = args.y_start
    y_index_end = args.y_end

    x_index_start = args.x_start
    x_index_end = args.x_end

    # Going through all the cells using for loops over the indices
    for y_index in range(y_index_start, y_index_end):
        for x_index in range(x_index_start, x_index_end):

            # Reading the image of the cell
            image_path = os.path.join(
                args.image_folder, "{0}_{1}.png".format(y_index, x_index)
            )

            if os.path.exists(image_path):
                rgb_dist = extract_rgb(img=Image.open(image_path))
                rgb_dist_df = pd.DataFrame(rgb_dist, columns=["r", "g", "b", "x", "y"])

                rgb_dist_df = rgb_dist_df.drop_duplicates(
                    subset=["r", "g", "b"], ignore_index=True
                )  # dropping duplicate rgb data values

                # Extracting the estimated and interpolated data value
                rgb_dist_df["data"] = rgb_dist_df.apply(
                    lambda row: interpolate_data(
                        r=row["r"],
                        g=row["g"],
                        b=row["b"],
                        df=rgb_scale_df,
                    ),
                    axis=1,
                )

                dist_store.append(
                    {
                        "x_index": x_index,
                        "y_index": y_index,
                        "data": np.mean(
                            rgb_dist_df["data"]
                        ),  # Since the image is of a complete cell, mean provides correct estimation
                    }
                )

    dist_store_df = pd.DataFrame(dist_store)

    # Reversing the y_index for better understanding
    dist_store_df["new_y_index"] = y_index_end - 1 - dist_store_df["y_index"]
    dist_store_df = dist_store_df.drop(columns=["y_index"])
    dist_store_df = dist_store_df.rename(columns={"new_y_index": "y_index"})

    # Storing the information extracted
    dist_store_df.to_csv(args.output_file)


def plot_rgb_subparser_func(args):
    """Driver function to plot the data values against indices

    Parameters
    ----------
    args.data_file: str
        DataFrame file storing data values against (x_index, y_index)
    args.upper_limit: float
        Upper Limit for scale
    args.lower_limit: float
        Lower Limit for scale
    args.data_label: str
        Label for property you extracted
    args.output_plot: str
        Plot file name
    """

    dist_store_df = pd.read_csv(args.data_file)

    if "x_index" and "y_index" not in dist_store_df.columns:
        raise RuntimeError(
            "Indices for plot are absent, you might have provided scale datafile!"
        )

    # Plotting the information extracted
    plot = plot_rgb(
        df=dist_store_df,
        upper_limit=args.upper_limit,
        lower_limit=args.lower_limit,
        label=args.data_label,
    )
    plot.savefig(args.output_plot, dpi=2000)
    plt.close()


def main():
    print(80 * "#")

    parser = argparse.ArgumentParser(
        prog="RGB_Digitization",
        description="""
        Python tool to form extract and plot rgb values from a required image
        """,
    )
    subparsers = parser.add_subparsers(help="subcommands of RGB Digitize")

    extract_scale_parser = subparsers.add_parser("scale")
    extract_indices_parser = subparsers.add_parser("index")
    plot_rgb_parser = subparsers.add_parser("plot")

    # -----------------------------------------------------------------
    # OPTIONS FOR THE extract_scale SUBPARSER
    # -----------------------------------------------------------------

    extract_scale_parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        help="Scale Image Path",
    )
    extract_scale_parser.add_argument(
        "-u",
        "--upper_limit",
        type=float,
        help="Upper Limit for scale",
    )
    extract_scale_parser.add_argument(
        "-l",
        "--lower_limit",
        type=float,
        help="Lower Limit for scale",
    )
    extract_scale_parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Dataframe file to store (r,g,b) vs data values",
    )
    extract_scale_parser.set_defaults(func=extract_scale_subparser_func)

    # -----------------------------------------------------------------
    # OPTIONS FOR THE extract_indices SUBPARSER
    # -----------------------------------------------------------------

    extract_indices_parser.add_argument(
        "-s",
        "--scale_file",
        type=str,
        help="Scale DataFrame file",
    )
    extract_indices_parser.add_argument(
        "--y_start",
        type=int,
        help="Start of Y-index for the images",
    )
    extract_indices_parser.add_argument(
        "--y_end",
        type=int,
        help="End of Y-index for the images",
    )
    extract_indices_parser.add_argument(
        "--x_start",
        type=int,
        help="Start of X-index for the images",
    )
    extract_indices_parser.add_argument(
        "--x_end",
        type=int,
        help="End of X-index for the images",
    )
    extract_indices_parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        help="Index Images Folder",
    )
    extract_indices_parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Dataframe file to store data values vs (x_index, y_index)",
    )
    extract_indices_parser.set_defaults(func=extract_indices_subparser_func)

    # -----------------------------------------------------------------
    # OPTIONS FOR THE plot_rgb SUBPARSER
    # -----------------------------------------------------------------

    plot_rgb_parser.add_argument(
        "-d",
        "--data_file",
        type=str,
        help="DataFrame file storing data values against (x_index, y_index)",
    )
    plot_rgb_parser.add_argument(
        "-u",
        "--upper_limit",
        type=float,
        help="Upper Limit for scale",
    )
    plot_rgb_parser.add_argument(
        "-l",
        "--lower_limit",
        type=float,
        help="Lower Limit for scale",
    )
    plot_rgb_parser.add_argument(
        "-b",
        "--data_label",
        type=str,
        help="Label for property extracted",
    )
    plot_rgb_parser.add_argument(
        "-o",
        "--output_plot",
        type=str,
        help="Plot file location",
    )
    plot_rgb_parser.set_defaults(func=plot_rgb_subparser_func)

    # Parse arguments and call the subparser functions
    args = parser.parse_args()
    args.func(args)

    print("All Done!")
    print(80 * "#")


if __name__ == "__main__":
    main()
