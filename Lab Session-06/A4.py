
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


file_paths = ['255_s.xlsx', '259_s.xlsx', '261_s.xlsx', '285_s.xlsx', '287_s.xlsx',
              '219_student.xlsx', '220_student.xlsx', '221_student.xlsx', '222_student.xlsx', '223_student.xlsx']


df_list = [pd.read_excel(file, sheet_name='Sheet1', engine='openpyxl') for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

# Function to bin a column
def bin_column(df, column, bins=4, strategy='uniform'):
    kbins = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    df[column] = kbins.fit_transform(df[[column]])
    return df

# Example usage
# Replace 'Start time' with the column you want to bin
df = bin_column(df, 'Start time', bins=4, strategy='uniform')
print(df['Start time'].value_counts())
