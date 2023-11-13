import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

# Calculate the correlation matrix
correlation_matrix = df.iloc[:, 1:-1].corr()

# Plot the heatmap using Plotly
fig = px.imshow(correlation_matrix, labels=dict(x="Tier", y="Tier", color="Correlation"), x=correlation_matrix.index, y=correlation_matrix.columns)
fig.update_layout(title='Correlation Matrix Heatmap', autosize=False, width=600, height=600)
fig.show()
