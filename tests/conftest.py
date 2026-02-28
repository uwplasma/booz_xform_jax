import os

# Ensure tests never try to open interactive windows.
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        # Matplotlib may be optional for some test runs.
        pass


def pytest_runtest_teardown(item):
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
