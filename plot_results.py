import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/mpi_benchmark_results.csv")

plt.figure(figsize=(10, 6))
for n in df["Matrix Size (N)"].unique():
    subset = df[df["Matrix Size (N)"] == n]
    plt.plot(subset["Processes"], subset["Time (s)"], marker='o', label=f'N={n}')

plt.title("MPI Matrix Multiplication Performance")
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/mpi_scalability_plot.png")
plt.show()
