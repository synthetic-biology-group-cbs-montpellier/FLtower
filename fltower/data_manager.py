import json
import os


def load_parameters(input_dir, parameters_file=None):
    """
    Load plotting parameters from a JSON file.

    Parameters
    ----------
    input_dir : str
        Directory where the default parameters file is stored.
    parameters_file : str or None
        Path to the parameters JSON file. If None, defaults to `parameters.json` in `input_dir`.

    Returns
    -------
    dict
        The loaded parameters.
    """
    # Determine file path
    file_path = (
        parameters_file
        if parameters_file
        else os.path.join(input_dir, "parameters.json")
    )

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"'parameters.json' file not found: {file_path}")

    # Load the JSON data
    with open(file_path, "r") as f:
        return json.load(f)
