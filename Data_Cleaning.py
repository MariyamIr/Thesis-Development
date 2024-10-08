import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#loading the dataset and checking initial few rows
df = pd.read_csv("dataset.csv")
print(df.head())  #looking at the first few rows
print(df.info())  #checking for nulls, data types, etc.

#checking for duplicates and removing them
duplicate_rows = df.duplicated().sum()  #counting duplicate rows
print(f'Duplicates: {duplicate_rows}')
df.drop_duplicates(inplace=True)  #dropping any duplicate rows

#generating basic statistics to understand the data
print(df.describe())  #getting an overview of numerical columns

#exploring unique values in categorical columns
categorical_columns = ['instance_events_type', 'scheduling_class', 'collection_type', 'event', 'failed']
for column in categorical_columns:
    unique_values = df[column].unique()
    print(f'Unique values in {column}: {unique_values}')

#plotting the distribution of 'assigned_memory' to check spread
plt.figure(figsize=(10, 6))
df['assigned_memory'].hist(bins=30)
plt.title('Histogram of Assigned Memory')
plt.xlabel('Assigned Memory')
plt.ylabel('Frequency')
plt.show()

#creating a countplot for 'event' to understand event distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='event', data=df)
plt.title('Event Count Plot')
plt.xticks(rotation=45)
plt.grid(False)
plt.show()

#generating correlation matrix to explore relationships between variables
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

#using a box plot to compare 'assigned_memory' across 'event'
plt.figure(figsize=(10, 6))
sns.boxplot(x='event', y='assigned_memory', data=df)
plt.title('Box Plot of Assigned Memory by Event')
plt.xticks(rotation=45)
plt.grid(False)
plt.show()

#handling missing values by initially filling numeric columns with mean values
missing_values = df.isnull().sum()
print("Missing values before handling:\n", missing_values)

#filling missing values in numerical columns with mean
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    df[column].fillna(df[column].mean(), inplace=True)

#trying filling missing categorical columns with mode (most frequent value)
for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

#checking missing values again after handling
missing_values_after = df.isnull().sum()
print("Missing values after handling:\n", missing_values_after)

#ensuring 'resource_request' is numeric and handling any errors in conversion
df['resource_request'] = pd.to_numeric(df['resource_request'], errors='coerce')

#generating statistics again to check any changes after handling missing data
print(df.describe())

#dropping unnecessary columns for analysis and keeping the required ones
columns_to_keep = ['instance_events_type', 'memory_accesses_per_instruction', 'failed', 'time']
df = df[columns_to_keep]

#checking the updated dataset
print(df.head())

#filling missing values in 'memory_accesses_per_instruction' with the mean
df['memory_accesses_per_instruction'].fillna(df['memory_accesses_per_instruction'].mean(), inplace=True)

#creating a new feature 'estimated_power_consumption' for potential analysis
df['estimated_power_consumption'] = df['memory_accesses_per_instruction'] * 100  #arbitrary calculation for draft

#saving the cleaned and feature-engineered dataset to a new CSV file
df.to_csv("processed_dataset.csv", index=False)

