import pandas as pd
from datetime import datetime, date, timedelta

# Load the Excel file
file_path = 'GLEX_NetJets.xlsx'
xls = pd.ExcelFile(file_path)

# Filter sheets that start with "N"
sheets_to_read = [sheet for sheet in xls.sheet_names if sheet.startswith("N")]

# Read sheets into a dictionary
df_dict = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in sheets_to_read}

# Define the columns to check for errors
columns_to_check1 = ['FLIGHT TIME', 'STD', 'ATD', 'STA', 'ORIG']
columns_to_check2 = ['STATUS']

# Define conversion functions
def convert_to_datetime(series):
    return series.astype(str).apply(
        lambda x: x.date() if isinstance(x, pd.Timestamp) else datetime.strptime(x, "%Y-%m-%d").date()
    )

def convert_to_timedelta(series):
    return series.astype(str).apply(
        lambda x: timedelta(hours=int(x.split(':')[0]), minutes=int(x.split(':')[1]))
        if pd.notna(x) and ':' in x else pd.NaT
    )

def convert_to_time(series):
    return series.astype(str).apply(
        lambda x: datetime.strptime(x, "%H:%M:%S").time()
        if pd.notna(x) and ':' in x else None
    )
# Function to clean each DataFrame
def clean_dataframe(df, tailsign):
    # Remove rows where any of the specified columns contain "—"
    for col in columns_to_check1:
        if col in df.columns:
            df = df[~df[col].astype(str).str.contains("—", na=True)]

    for col in columns_to_check2:
        if col in df.columns:
            df = df[~df[col].astype(str).str.contains('Diverted', na=False)]

    # Convert column types
    df['DATE'] = convert_to_datetime(df['DATE'])
    df['FLIGHT TIME'] = convert_to_timedelta(df['FLIGHT TIME'])
    for col in ['STD', 'ATD', 'STA']:
        if col in df.columns:
            df[col] = convert_to_time(df[col])

    df['TAIL'] = tailsign

    return df

# Clean all DataFrames in df_dict
for key in df_dict.keys():
    original_count = len(df_dict[key])
    df_dict[key] = clean_dataframe(df_dict[key], key)
    cleaned_count = len(df_dict[key])
    print("Cleaned sheet %s: removed %d rows, remaining %d rows" % (key, original_count - cleaned_count, cleaned_count))


# Define function to create transformed DataFrame
def transform_dataframe(df):

    # Safe combination of DATE and ATD into DEP
    def safe_combine(date, time_obj):
        try:
            return datetime.combine(date, time_obj) if pd.notna(date) and pd.notna(time_obj) else None
        except Exception as e:
            print("Error combining date and time: %s" % e)
            return None

    df['DEP'] = df.apply(lambda row: safe_combine(row['DATE'], row['ATD']), axis=1)

    # Create ARR column as DEP + FLIGHT TIME (timedelta)
    df['ARR'] = df.apply(lambda row: row['DEP'] + row['FLIGHT TIME'] if pd.notna(row['DEP']) and pd.notna(row['FLIGHT TIME']) else None, axis=1)

    # Create ROUTE as "ORIG_DEST"
    df['ROUTE'] = df.apply(lambda row: "%s_%s" % (row['ORIG'], row['DEST']) if pd.notna(row['ORIG']) and pd.notna(row['DEST']) else None, axis=1)

    # Compute Turnaround Time (TAT) as the gap between ARR and the next row's DEP
    df['TAT'] = df['DEP'].shift(+1) - df['ARR']

    # Identify inconsistencies: if previous DEST is not equal to next ORIG, set TAT to NaT
    df.loc[df['DEST'].shift(-1) != df['ORIG'], 'TAT'] = pd.NaT

    # Select required columns for the final DataFrame
    df_transformed = df[['ORIG', 'DEST', 'ROUTE', 'DEP', 'ARR', 'FLIGHT TIME', 'TAT', 'TAIL']].rename(columns={'FLIGHT TIME': 'FTIME'})

    return df_transformed

# Process all sheets and concatenate into one DataFrame
df_list = [transform_dataframe(df) for df in df_dict.values()]
df_final = pd.concat(df_list, ignore_index=True)

# Print summary
print("Final DataFrame created with %d rows" % len(df_final))

df_final.to_excel('GLEX_NetJets_Transformed.xlsx', index=False)