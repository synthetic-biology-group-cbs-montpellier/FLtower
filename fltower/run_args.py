# -*- coding: utf-8 -*-

import os
from argparse import ArgumentParser


def parse_run_args(command_line_arguments=None):
    """Parse run arguments

    Parameters
    ----------
    command_line_arguments : List[str]
        Used to give inputs for the runtime when you call this function like a module.
        For example, to test the FLtower run from tests folder.

    Returns
    -------
    ArgumentParser.args
        An accessor of run arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-I",
        "--input",
        type=str,
        default=os.getcwd(),
        help="Folder path with input data.\nDEFAULT: Current directory",
    )
    parser.add_argument(
        "-O",
        "--output",
        type=str,
        default=os.getcwd(),
        help="Folder path for output data.\nDEFAULT: Current directory",
    )
    parser.add_argument(
        "-P",
        "--parameters",
        type=str,
        default=None,
        help="Path of the parameters.json file.\nDEFAULT: FLtower expect this file inside your current directory",
    )
    parser.add_argument(
        "--exp1_dir",
        help="Directory containing FCS files for experiment 1",
    )

    parser.add_argument(
        "--exp2_dir",
        help="Directory containing FCS files for experiment 2",
    )
    parser.add_argument(
        "--exp1_label",
        default="EXP1",
        help="Label for experiment 1 (e.g. 6h, Day1, Control)",
    )

    parser.add_argument(
        "--exp2_label",
        default="EXP2",
        help="Label for experiment 2 (e.g. 24h, Day2, Treated)",
    )

    return parser.parse_args(command_line_arguments)
