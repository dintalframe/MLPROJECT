import pandas as pd

# Function to remove rows with any empty data, filter out specific values in Batch_ID, and remove extreme outliers
def remove_empty_rows_and_outliers(input_csv_path, output_csv_path):
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv_path)

        # Drop rows with any empty cells
        df_cleaned = df.dropna()

        # Remove rows with 'cal.', 'calibration', or 'cal' in the 'Batch_ID' column (case insensitive)
        df_cleaned = df_cleaned[~df_cleaned['Batch_ID'].str.contains(r'\b(cal\.|calibration|cal)\b', case=False, na=False)]

        # Function to remove extreme outliers using the IQR method or capping
        def remove_or_cap_outliers(df, column, lower_quantile=0.03, upper_quantile=0.97, absolute_upper_bound=None):
            lower_bound = df[column].quantile(lower_quantile)
            upper_bound = df[column].quantile(upper_quantile)
            if absolute_upper_bound is not None:
                upper_bound = min(upper_bound, absolute_upper_bound)
            # Cap outliers to the nearest boundary
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            return df

        # List of numeric columns to check for outliers and specific adjustments
        numeric_columns = [
            ('Efficiency_(percent)', None),
            ('Rsh_(ohm-cm2)', 100000),  # Cap Rsh at 100,000 for solar cell standards
            ('Rs_(ohm-cm2)', None),
            ('n_at_1_sun', None),
            ('n_at_1/10_suns', None),
            ('Jo2_(A/cm2)', None)
        ]

        # Remove or cap outliers for each numeric column
        for col, absolute_upper_bound in numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned = remove_or_cap_outliers(df_cleaned, col, absolute_upper_bound=absolute_upper_bound)

        # Save the cleaned DataFrame to the output CSV file
        df_cleaned.to_csv(output_csv_path, index=False)
        print(f"Empty rows, calibration-related rows, and extreme outliers removed or capped. Cleaned data saved to {output_csv_path}")
    except Exception as e:
        print(f"Error processing {input_csv_path}: {e}")

if __name__ == "__main__":
    # Set the path to the input CSV file containing compiled features
    input_csv_path = "/mnt/c/Users/PVRL-01/Documents/Donald Intal/mlproj/output/compiled_features.csv"
    # Set the path for the output CSV file
    output_csv_path = "/mnt/c/Users/PVRL-01/Documents/Donald Intal/mlproj/output/compiled_features_cleaned.csv"

    # Call the function to remove empty rows, unwanted Batch_ID rows, and extreme outliers
    remove_empty_rows_and_outliers(input_csv_path, output_csv_path)
