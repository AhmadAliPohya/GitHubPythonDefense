import pandas as pd

# Load the data
df = pd.read_excel('GLEX_NetJets_Transformed_overallstats.xlsx', sheet_name='ForSim')

# Convert relevant columns to numeric, forcing errors to NaN
df[["DIST", "TIME", "FREQ"]] = df[["DIST", "TIME", "FREQ"]].apply(pd.to_numeric, errors='coerce')

# Drop rows where any critical column has NaN values
df_cleaned = df.dropna(subset=["DIST", "TIME", "FREQ"]).copy()

# Find destinations that do not appear as origins
missing_origins = df_cleaned.loc[~df_cleaned["DEST"].isin(df_cleaned["ORIG"])].copy()

# Create new rows for missing return flights
missing_origins["ORIG"], missing_origins["DEST"] = missing_origins["DEST"], missing_origins["ORIG"]
missing_origins["ROUTE"] = missing_origins["ORIG"] + "_" + missing_origins["DEST"]

# Print the return flights that are being added
if not missing_origins.empty:
    print("Added return flights:")
    print(missing_origins[["ORIG", "DEST", "ROUTE"]].to_string(index=False))
else:
    print("No return flights were missing.")

# Append missing return flights to the cleaned DataFrame
df_final = pd.concat([df_cleaned, missing_origins], ignore_index=True)

# Save or display the final DataFrame
df_final.to_csv('flight_schedule.csv', index=False)
print("Updated flight schedule saved to 'updated_flight_schedule.csv'.")
