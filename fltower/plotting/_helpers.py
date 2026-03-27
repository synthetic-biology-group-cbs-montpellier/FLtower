"""Shared helpers for plotting modules."""

# Mapping labels for channels
LABEL_MAP = {
    "BL1-H": "GFP",
    "YL2-H": "RFP",
    "SSC-A": "SSC-A",
    "SSC-H": "SSC-H",
    # Add more mappings here as needed
}


def get_label(param):
    return LABEL_MAP.get(param, param)
