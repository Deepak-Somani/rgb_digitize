from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np


def extract_rgb(img: Image, color_scale: bool = False) -> list:
    """Returns a list of tuple with rgb and coordinate values
    for each of the pixel in the provided image

    Parameters
    ----------
    img: Image
        Image input
    color_scale: bool
        if the image provided is a color scale or not

    Returns
    -------
    rgb_values: list of tuples
        - [(r, g, b, x, y)]: [(int, int, int, float, float)]

    """

    img_rgb = img.convert("RGB")
    width, height = img.size

    if color_scale:
        x_range = [int(width / 2)]
    else:
        x_range = range(width)

    rgb_values = []

    for y in range(height):
        for x in x_range:
            r, g, b = img_rgb.getpixel((x, y))

            # Some constraints to avoid black, grey and white colors
            if (
                (r, g, b) != (254, 254, 254)
                and (r, g, b) != (255, 255, 255)
                and (r, g, b) != (0, 0, 0)
                and (r, g, b) != (0, 1, 6)
                # and (r != g and g != b and b != r)
                and r + g + b >= 60
            ):
                rgb_values.append((r, g, b, x, y))

    return rgb_values


def find_closest_points(r: int, g: int, b: int, df: pd.DataFrame) -> list:
    """Returns the index of two closest color data points (euclidean distance)
    from the input dataframe based on provided r,g,b color data values

    Parameters
    ----------
    r, g, b: int, int, int
        rgb data values
    df: pd.DataFrame
        pandas dataframe holding the color-data information for the scale of colormap
        Column names and types for the dataframe are:
        - r, g, b : int, int, int
            r,g,b values
        - x, y : float, float
            coordinates

    Returns
    -------
    closest_indices: list
        two closest indices to the given r,g,b value from the dataframe of scale
    """

    # Calculate Euclidean distances from the given point to all points in the DataFrame
    distances = np.sqrt((df["r"] - r) ** 2 + (df["g"] - g) ** 2 + (df["b"] - b) ** 2)

    # Find the indices of the two closest points
    closest_indices = np.argsort(distances)[:2]

    return closest_indices


def interpolate_data(r, g, b, df) -> float:
    """Returns the estimated data value corresponding to the
    provided r,g,b color-data value based on the color-data of scale of the colormap
    stored in the provided pandas DataFrame

    Parameters
    ----------
    r, g, b: int, int, int
        rgb data values
    df: pd.DataFrame
        pandas dataframe holding the color-data information for the scale of colormap
        Column names and types for the dataframe are:
        - r, g, b : int, int, int
            r,g,b values
        - x, y : float, float
            coordinates

    Returns
    -------
    interpolated_value: float
        estimated value by linear interpolation
    """

    # Find the indices of the two closest points
    closest_indices = find_closest_points(r, g, b, df)

    # Get the coordinates and data values of the two closest points
    point1 = df.iloc[closest_indices[0]]
    point2 = df.iloc[closest_indices[1]]

    # Calculate the weights based on the Euclidean distance
    total_distance = np.sqrt(
        (point2["r"] - point1["r"]) ** 2
        + (point2["g"] - point1["g"]) ** 2
        + (point2["b"] - point1["b"]) ** 2
    )
    weight1 = (
        np.sqrt(
            (r - point1["r"]) ** 2 + (g - point1["g"]) ** 2 + (b - point1["b"]) ** 2
        )
        / total_distance
    )
    weight2 = (
        np.sqrt(
            (r - point2["r"]) ** 2 + (g - point2["g"]) ** 2 + (b - point2["b"]) ** 2
        )
        / total_distance
    )

    # Perform linear interpolation
    interpolated_value = (point1["data"] * weight2 + point2["data"] * weight1) / (
        weight1 + weight2
    )

    return interpolated_value


def plot_rgb(
    df: pd.DataFrame, upper_limit: float, lower_limit: float, label: str
) -> Figure:
    """Returns a distribution plot

    Parameters
    ----------
    df: pd.DataFrame
        pandas dataframe holding the color-data information for the required cells
        Column names and types for the dataframe are:
        - r, g, b : int, int, int
            r,g,b values
        - x_index, y_index : int, int
            user-provided indices

    upper_limit: float
        upper limit for the data value
    lower_limit: float
        lower limit for the data value
    label: str
        data label
    """

    f = plt.subplots()
    plt.figure(figsize=(4, 4))
    plt.scatter(
        df["x_index"],
        df["y_index"],
        c=df["data"],
        cmap="jet",
    )
    cbar = plt.colorbar()
    plt.clim(lower_limit, upper_limit)
    cbar.set_label(label)
    plt.xlabel("x")
    plt.ylabel("y")

    return plt.gcf()
