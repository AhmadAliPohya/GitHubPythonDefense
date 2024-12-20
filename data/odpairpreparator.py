import pandas as pd
import numpy as np
import configparser as cp

# Read the config file and CSV
config = cp.ConfigParser()
config.read('..\\config.ini')
df_raw = pd.read_csv('SCHEDULES_ALL_ALL_DIR_2023JAN_2023DEC.csv')

# Step 1: Remove the last 5% of rows based on cumulative frequency
cutoff95idx = int(np.where(np.cumsum(df_raw['DepCount'] / df_raw['DepCount'].sum()) > 0.95)[0][0])
df_raw = df_raw.iloc[:cutoff95idx]

# Step 2: Create the new DataFrame
df_new = pd.DataFrame()
df_new['ORIG'] = df_raw['Origin Airport']
df_new['DEST'] = df_raw['Destination Airport']
df_new['ROUTE'] = df_new['ORIG'] + '_' + df_new['DEST']
df_new['DIST'] = df_raw['Distance (km)']
df_new['TIME'] = (0.0717 * df_new['DIST'] + 24.1) / 60

# Step 3: Filter rows where airports in ORIG and DEST are valid
valid_airports = set(df_new['ORIG']).intersection(set(df_new['DEST']))
df_new = df_new[df_new['ORIG'].isin(valid_airports) & df_new['DEST'].isin(valid_airports)]

# Step 4: Normalize the frequency column
df_new['FREQ'] = df_raw['DepCount'] / df_raw['DepCount'].sum()
df_new['FREQ'] = df_new['FREQ'] / df_new['FREQ'].sum()

# Step 5: Save the filtered DataFrame
df_new.to_csv('data.csv', index=False)

