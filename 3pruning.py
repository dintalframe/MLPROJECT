import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to clean the dataset and save in the "output" folder
def remove_empty_rows_and_outliers():
    try:
        # Define input and output folders
        input_folder = "outputunclean"
        output_folder = "output"
        input_file_name = "compiled_features.csv"
        input_csv_path = os.path.join(input_folder, input_file_name)

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Read the CSV file
        logging.info(f"Reading input CSV file from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        initial_row_count = len(df)
        logging.info(f"Initial row count: {initial_row_count}")

        # Drop rows with any empty cells
        df_cleaned = df.dropna()
        logging.info(f"Rows with missing values removed. Remaining rows: {len(df_cleaned)}")

        # Remove rows with 'cal.', 'calibration', 'cal', or 'calibrate' in the 'Batch_ID' column (case insensitive)
        df_cleaned = df_cleaned[~df_cleaned['Batch_ID'].str.contains(r'\b(cal\.|calibration|cal|calibrate)\b', case=False, na=False)]
        logging.info(f"Rows with unwanted 'Batch_ID' values removed. Remaining rows: {len(df_cleaned)}")

        # Function to remove extreme outliers using the IQR method or capping
        def remove_or_cap_outliers(df, column, lower_quantile=0.03, upper_quantile=0.97, absolute_upper_bound=None):
            lower_bound = df[column].quantile(lower_quantile)
            upper_bound = df[column].quantile(upper_quantile)
            if absolute_upper_bound is not None:
                upper_bound = min(upper_bound, absolute_upper_bound)
            # Cap outliers to the nearest boundary
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            logging.info(f"Outliers capped for column '{column}': lower_bound={lower_bound}, upper_bound={upper_bound}")
            return df

        # Function to handle negative values
        def handle_negative_values(df, column):
            negative_count = (df[column] < 0).sum()
            if negative_count > 0:
                logging.warning(f"Column '{column}' has {negative_count} negative values. Replacing with 0.")
                df[column] = df[column].clip(lower=0)  # Replace negative values with 0
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

        # Check and handle negative values in numeric columns
        for col, _ in numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned = handle_negative_values(df_cleaned, col)

        # Save the cleaned DataFrame to the "output" folder
        output_file_path = os.path.join(output_folder, "compiled_features_cleaned.csv")
        df_cleaned.to_csv(output_file_path, index=False)
        logging.info(f"Cleaned data saved to {output_file_path}")

        # Generate a summary report
        final_row_count = len(df_cleaned)
        logging.info(f"Final row count: {final_row_count}")
        logging.info(f"Total rows removed: {initial_row_count - final_row_count}")
    except Exception as e:
        logging.error(f"Error processing the dataset: {e}")

if __name__ == "__main__":
    # Call the function to remove empty rows, unwanted Batch_ID rows, and extreme outliers
    remove_empty_rows_and_outliers()
