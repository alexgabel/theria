import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stress_rel_diff_triton.csv")

plt.figure(figsize=(4,3))
plt.plot(df["inner_steps"], df["rel_diff_full_vs_fo"], marker="o")
plt.xscale("log")
plt.xlabel("Inner steps")
plt.ylabel("‖FULL − FO‖ / ‖FULL‖")
plt.title("Second-order signal grows with inner-loop depth")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("stress_rel_diff_triton.png", dpi=200)
plt.show()