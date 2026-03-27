"""FCS file reading utilities."""

import logging

import fcsparser

logger = logging.getLogger("fltower")


def read_fcs(file_path):
    try:
        meta, data = fcsparser.parse(file_path, reformat_meta=True)
        return data, list(data.columns)
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e} (type: {type(e).__name__})")
        return None, []
