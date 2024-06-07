# RGB_Digitize

Code to digitize and extract the value of different cells of colormap

**RGB_Digitize** can help you extract the data value from a colormap.
The procedure to do so can be acknowledge by looking over the "example" folder.
If you want to extract the data values of certain points in "example/images/dist_plot.png" with a given scale of "example/images/dist_scale.png", you can do so by following below procedure:

1. Extract the scale values using `extract_scale_parser`: Have a look at "example/data/dist_scale.csv" file.
2. Manually slice up your colormap according to "(x-index)_(y-index).png". Have a look at "example/images" folder.
3. Feed these images by "x-index" and "y-index" to `extract_indices_parser` to create a dataframe file that will contain the extracted data values for you. Have a look at "examples/data/dist_data.csv" file.
4. If you wish to plot the colormap for verification, you can use `plot_rgb_parser`. Have a look at "examples/plots/dist_plot.png" file.