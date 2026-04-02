# Import Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load Dataset
df = pd.read_csv("dirty_cafe_sales.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

# Task 1
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Task 2
df = df.replace("UNKNOWN", np.nan)


df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
df['Total Spent'] = pd.to_numeric(df['Total Spent'], errors='coerce')


df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')


num_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)


cat_cols = ['Item', 'Payment Method', 'Location']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Task 3
Q1 = df['Total Spent'].quantile(0.25)
Q3 = df['Total Spent'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Total Spent'] < lower) | (df['Total Spent'] > upper)]
print("\nNumber of Outliers:")
print(len(outliers))


df_no_outliers = df[(df['Total Spent'] >= lower) & (df['Total Spent'] <= upper)]
print("\nDataset Shape After Removing Outliers:")
print(df_no_outliers.shape)

# Task 4
minmax = MinMaxScaler()
df_minmax = df_no_outliers.copy()
df_minmax[num_cols] = minmax.fit_transform(df_minmax[num_cols])
print("\nMin-Max Normalization Applied")

zscore = StandardScaler()
df_zscore = df_no_outliers.copy()
df_zscore[num_cols] = zscore.fit_transform(df_zscore[num_cols])
print("Z-score Normalization Applied")

# Task 5
corr = df_no_outliers[num_cols].corr()
print("\nCorrelation Matrix:")
print(corr)

# Task 6

imputer = SimpleImputer(strategy='median')
X_num = imputer.fit_transform(df_no_outliers[num_cols])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_num)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

