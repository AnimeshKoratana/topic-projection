import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()

data = pd.read_csv("metrics.csv")

data = data.pivot(index="Dataset Size", columns="K", values="Value")

sns.heatmap(data, fmt="g", linewidths=.5)
plt.savefig("heatmap.pdf")
plt.show()
