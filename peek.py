import pandas as pd

# Read the Excel file instead of a CSV
df = pd.read_excel('External_Cibil_Dataset.xlsx')

# Print the columns
print(df.columns.tolist())