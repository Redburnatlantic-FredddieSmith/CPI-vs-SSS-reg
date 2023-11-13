import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import numpy as np

# Load your Excel data
df = pd.read_excel('CPI regression.xlsx', sheet_name='Names')

# Ensure numeric data types for relevant columns
df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
df.iloc[:, 1:-1] = df.iloc[:, 1:-1].apply(lambda x: pd.to_numeric(x, errors='coerce') / 100.0)

# Perform linear regression for each tier
results = {}
for column in df.columns[1:-1]:  # Exclude only the 'Year' and 'CPI' columns
    x = df['CPI']
    y = df[column]

    # Remove missing values
    non_missing_mask = ~pd.concat([x, y], axis=1).isnull().any(axis=1)
    x = x[non_missing_mask]
    y = y[non_missing_mask]

    # Check if there are valid data points
    if len(x) > 0 and len(y) > 0:
        # Add a constant term to the independent variable (CPI)
        x = sm.add_constant(x)

        # Fit the linear regression model
        model = sm.OLS(y, x).fit()

        # Store the results
        results[column] = {
            'coef_CPI': model.params['CPI'],
            'intercept': model.params['const'],
            'r_squared': model.rsquared
        }
    else:
        print(f"No valid data points for {column}")

# Create a DataFrame from the results
results_df = pd.DataFrame(results).T

# Display the regression results
print("Regression Results:")
print(results_df)

# Calculate the correlation matrix, including correlation to CPI
correlation_matrix = df.corr()
np.fill_diagonal(correlation_matrix.values, np.nan)

# Exclude the "Year" column from the heatmap
correlation_matrix = correlation_matrix.drop("Year", axis=0)
correlation_matrix = correlation_matrix.drop("Year", axis=1)

# Plot the heatmap using Plotly
fig = px.imshow(correlation_matrix, labels=dict(x="Variable", y="Variable", color="Correlation"), x=correlation_matrix.index, y=correlation_matrix.columns)
fig.update_layout(title='Correlation Matrix Heatmap', autosize=False, width=800, height=800)

# Add correlation values as text annotations
for i, row in enumerate(correlation_matrix.index):
    for j, col in enumerate(correlation_matrix.columns):
        if not np.isnan(correlation_matrix.loc[row, col]):
            fig.add_annotation(
                x=i,
                y=j,
                text=f"{correlation_matrix.loc[row, col]:.2f}",
                showarrow=False,
                font=dict(color='black', size=10),
            )

# Save the heatmap as a PNG file
fig.write_image('correlation_heatmap.png')

# Print a message about where to find the PNG file
print("Correlation matrix heatmap saved as 'correlation_heatmap.png'. Open the file to view the plot.")






