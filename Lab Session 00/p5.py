# Question-10 
import pandas as pd
import numpy as np
file_path="City_State_PIN.xlsx"

df = pd.read_excel(file_path)

if {"City", "State", "PIN Code"}.issubset(df.columns):
    # Create a new column with "City, State"
    df["City, State"] = df["City"] + ", " + df["State"]

    # Write the updated DataFrame back to the Excel file
    updated_file_path = "New_City_Details.xlsx"  # Specify the output file path
    df.to_excel(updated_file_path, index=False)
    print(f"Updated Excel file saved as: {updated_file_path}")
else:
    print("Ensure the columns 'City', 'State', and 'PIN Code' exist in your Excel file.")