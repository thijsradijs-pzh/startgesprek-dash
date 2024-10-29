# utils.py

# Constants
CSV_FOLDER_PATH = './app_data/'
COLORMAP = 'viridis'
VIEW_STATE = {
    'longitude': 4.390,
    'latitude': 51.891,
    'zoom': 9
}

def clean_dataset_name(name):
    """Replaces underscores with spaces and capitalizes for cleaner display."""
    return ' '.join(word.capitalize() for word in name.replace('_', ' ').split())