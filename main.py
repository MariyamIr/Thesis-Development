import pandas as pd

#loading dataset
df = pd.read_csv("dataset.csv")

#checking for duplicates and dropping them
df.drop_duplicates(inplace=True)

#filling missing values in numerical columns with mean
df['memory_accesses_per_instruction'].fillna(df['memory_accesses_per_instruction'].mean(), inplace=True)

#keeping only necessary columns
columns_to_keep = ['instance_events_type', 'memory_accesses_per_instruction', 'failed', 'time']
df = df[columns_to_keep]

#creating a new feature 'estimated_power_consumption'
df['estimated_power_consumption'] = df['memory_accesses_per_instruction'] * 100  #example calculation

#saving the processed dataset to a new CSV file
df.to_csv("processed_dataset.csv", index=False)
