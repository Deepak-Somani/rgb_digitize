from rgb_digitize.rgb_tools import (
    extract_rgb,
    interpolate_data,
)
from sklearn.metrics import mean_absolute_error
from PIL import Image
import pandas as pd
import numpy as np


def almost_equal(x, y, threshold=0.01):
    return mean_absolute_error(x, y) <= threshold


def test_rgb_extract():
    """
    This function can be used to verify the module with any scale images of train and test
    as long as they correspond to same colormap scale.

    Here, the images used for train and test scale are: "train_scale.png" and "test_scale.png".
    They both are matplotlib generated "jet" colormap scales.
    Test image: "test_scale.png" is compressed in y-direction.

    Any value of upper and lower limit can be chosen,
    here u=1, and l=-1 are chosen.
    """

    # can be random
    upper_limit = 1
    lower_limit = -1

    train_scale = extract_rgb(img=Image.open("train_scale.png"), color_scale=True)
    train_scale_df = pd.DataFrame(train_scale, columns=["r", "g", "b", "x", "y"]).drop(
        columns=["x", "y"]
    )
    train_scale_df["data"] = np.linspace(
        upper_limit, lower_limit, num=train_scale_df.shape[0]
    )
    train_scale_df = train_scale_df.drop_duplicates(
        subset=["r", "g", "b"], ignore_index=True
    )

    test_scale = extract_rgb(img=Image.open("test_scale.png"), color_scale=True)
    test_scale_df = pd.DataFrame(test_scale, columns=["r", "g", "b", "x", "y"]).drop(
        columns=["x", "y"]
    )
    test_scale_df["data"] = np.linspace(
        upper_limit, lower_limit, num=test_scale_df.shape[0]
    )
    test_scale_df = test_scale_df.drop_duplicates(
        subset=["r", "g", "b"], ignore_index=True
    )

    test_scale_df["calc_data"] = test_scale_df.apply(
        lambda row: interpolate_data(
            r=row["r"],
            g=row["g"],
            b=row["b"],
            df=train_scale_df,
        ),
        axis=1,
    )

    Y = np.array(test_scale_df["data"].tolist())
    Y_predicted = np.array(test_scale_df["calc_data"].tolist())

    assert almost_equal(x=Y, y=Y_predicted)
