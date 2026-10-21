
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = PROJECT_ROOT / "results" / "model_comparison.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if not RESULTS_PATH.exists():
    raise FileNotFoundError(
        "Comparison file not found. Run compare_models.py first."
    )

results_df = pd.read_csv(RESULTS_PATH)

# Plot accuracy and F1-score
ax = results_df.plot(
    x="model",
    y=["accuracy", "f1_score"],
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=15,
)

ax.set_title("Sentiment Classification Model Comparison")
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Score")
ax.legend(["Accuracy", "F1-score"])
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

output_path = FIGURES_DIR / "model_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Chart saved to: {output_path}")
