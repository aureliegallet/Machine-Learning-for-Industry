import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./climate_data.csv")
data_to_inspect = data[["DR", "RH", "RHX"]]
data_to_inspect["TOT"] = data_to_inspect["RH"] * data_to_inspect["DR"]
data_to_inspect["INT"] = data_to_inspect["RH"] / data_to_inspect["DR"]
transformed_data = data_to_inspect
transformed_data = transformed_data[["DR", "INT", "RHX"]]
# transformed_data = data_to_inspect[(data_to_inspect > 0)]
print(transformed_data)

figure, axis = plt.subplots(1, 5, figsize=(15, 5))  # Adjusted figure size for clarity

# Plotting each column
for i, column in enumerate(transformed_data.columns):
    axis[i].scatter(range(len(transformed_data[column])), transformed_data[column])
    axis[i].set_title("Daily values for the indicator " + column)
    axis[i].set_xlabel("Day")
    axis[i].set_ylabel(column)

# Show the entire figure
plt.tight_layout()  # Adjusts the layout to avoid overlap
plt.show()

print(transformed_data.corr(method="pearson"))

# plt.matshow(transformed_data.corr(method="pearson"))
# plt.show()
