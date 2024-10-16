# utils.py

import matplotlib.pyplot as plt

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'magma'
VIEW_STATE = {
    'longitude': 4.390,
    'latitude': 51.891,
    'zoom': 9
}

# Apply color mapping
def apply_color_mapping(df, value_column, colormap):
    """Applies a color map to a specified column of a DataFrame."""
    norm = plt.Normalize(vmin=df[value_column].min(), vmax=df[value_column].max())
    colormap_func = plt.get_cmap(colormap)
    df['color'] = df[value_column].apply(
        lambda x: [int(c * 255) for c in colormap_func(norm(x))[:3]]
    )  # Get RGB values

# Helper function to clean dataset names
def clean_dataset_name(name):
    """Replaces underscores with spaces and capitalizes for cleaner display."""
    return ' '.join(word.capitalize() for word in name.replace('_', ' ').split())
