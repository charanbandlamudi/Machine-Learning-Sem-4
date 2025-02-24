# Import necessary libraries
import pandas as pd
import statistics
from google.colab import files

# Upload the Excel file
uploaded = files.upload()

# Get the uploaded filename
file_name = list(uploaded.keys())[0]

# Load the Excel file
xls = pd.ExcelFile(file_name)

# Load the "IRCTC Stock Price" sheet
stock_data = pd.read_excel(xls, sheet_name="IRCTC Stock Price")

# Calculate Mean & Variance of Price Data (Column D)
price_mean = statistics.mean(stock_data.iloc[:, 3])  # Column D (Index 3)
price_variance = statistics.variance(stock_data.iloc[:, 3])

print(f"Mean of Price Data: {price_mean}")
print(f"Variance of Price Data: {price_variance}")
# Convert 'Date' column to datetime format
stock_data["Date"] = pd.to_datetime(stock_data["Date"])

# Extract 'Day of Week' from Date
stock_data["Day"] = stock_data["Date"].dt.day_name()

# Filter data for Wednesdays
wednesday_data = stock_data[stock_data["Day"] == "Wednesday"]

# Calculate mean for Wednesdays
wednesday_mean = statistics.mean(wednesday_data.iloc[:, 3])

print(f"Mean for Wednesdays: {wednesday_mean}")
print(f"Difference from Population Mean: {abs(price_mean - wednesday_mean)}")

# Filter data for April
april_data = stock_data[stock_data["Date"].dt.month == 4]

# Calculate mean for April
april_mean = statistics.mean(april_data.iloc[:, 3])

print(f"Mean for April: {april_mean}")
print(f"Difference from Population Mean: {abs(price_mean - april_mean)}")

# Probability of Making a Loss
loss_probability = sum(stock_data.iloc[:, 8] < 0) / len(stock_data)  # Column I (Index 8)

print(f"Probability of Making a Loss: {loss_probability}")

# Probability of Making a Profit on Wednesday
profit_wednesday = sum(wednesday_data.iloc[:, 8] > 0) / len(wednesday_data)

print(f"Probability of Making a Profit on Wednesday: {profit_wednesday}")

# Total number of profit days
profit_days = sum(stock_data.iloc[:, 8] > 0)

# Conditional Probability of Profit Given it's Wednesday
conditional_prob = profit_wednesday * (len(wednesday_data) / len(stock_data))

print(f"Conditional Probability of Profit Given it's Wednesday: {conditional_prob}")
