from pathlib import Path

from r1.reporting import summarize_results


if __name__ == "__main__":
    summarize_results(Path("results"))
