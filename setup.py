from setuptools import setup, find_packages


def read_version():
    with open("fltower/__version__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="fltower",
    version=read_version(),
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "fltower=fltower.main_fltower:main",
        ],
    },
)
