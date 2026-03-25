import matplotlib

# Force non-interactive backend before any other matplotlib import.
# This avoids GUI overhead and reduces memory usage during tests.
matplotlib.use("Agg")
