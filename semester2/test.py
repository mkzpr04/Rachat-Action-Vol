import matplotlib.pyplot as plt
import numpy as np

# Data for the main asset classes
categories_capitalization = ["Stocks", "Bonds", "Cryptocurrencies"]
values_capitalization = [100_000, 130_000, 2_000]  # In billions of dollars

categories_flows = ["Forex", "Derivatives"]
values_flows = [7_500, 500_000]  # Daily Forex volume, derivatives notional in billions

# Create the figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Bar chart for capitalization markets
x_pos = np.arange(len(categories_capitalization))
bars = ax1.bar(x_pos, values_capitalization, color=['blue', 'green', 'purple'], alpha=0.7)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(categories_capitalization)
ax1.set_yscale("log")  # Logarithmic scale to better visualize differences
ax1.set_ylabel("Value (billion dollars)")
ax1.set_title("Capitalization Markets")

# Add annotations for the values in billions
for i, value in enumerate(values_capitalization):
    ax1.text(i, value + 5000, f'{value:,}', ha='center', va='bottom', fontsize=10)

# Bubble chart for flow markets
scatter_forex = ax2.scatter([0], [values_flows[0]], s=values_flows[0] / 50, alpha=0.5, color='red')
scatter_derivatives = ax2.scatter([1], [values_flows[1]], s=values_flows[1] / 50, alpha=0.5, color='orange')

# Add annotations for the values in billions in the flow markets chart
for i, value in enumerate(values_flows):
    ax2.text(i, value + 100_000, f'{value:,}', ha='center', va='bottom', fontsize=10)

# Configure the axes and titles of the second chart
ax2.set_xticks([0, 1])
ax2.set_xticklabels(categories_flows)
ax2.set_yscale("log")
ax2.set_ylabel("Value (billion dollars)")
ax2.set_title("Flow Markets")

# Legend with reduced symbol size
ax2.legend([scatter_forex, scatter_derivatives], ['Forex', 'Derivatives'], markerscale=0.1)

# Add the grid
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

# Display
plt.tight_layout()
plt.show()
