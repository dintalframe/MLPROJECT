import os
import pandas as pd
from tqdm import tqdm

# Function to extract specific columns from each CSV file and compile them into one CSV
def extract_features_and_compile(input_folders, output_csv_path):
    # Define the columns to extract
    columns_to_extract = [
        "Cell_Area_(cm2)", "Batch_ID", "Sample_ID", "Efficiency_(percent)", 
        "Rsh_(ohm-cm2)", "Rs_(ohm-cm2)", "n_at_1_sun", "n_at_ 1/10_suns", "Jo2_(A/cm2)",
        "Isc_(A)", "Voc_(V)", "FF_(percent)"
    ]

    # Initialize an empty DataFrame to store extracted data
    compiled_data = pd.DataFrame()

    # Get the list of all CSV files in the input folders
    csv_files = []
    for input_folder in input_folders:
        csv_files.extend([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])

    # Create a progress bar for the overall process
    with tqdm(total=len(csv_files), desc="Extracting Features from CSV Files", unit="file") as pbar:
        # Iterate over all CSV files in the input folders
        for csv_path in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)

                # Extract only the columns that are present in the DataFrame
                available_columns = [col for col in columns_to_extract if col in df.columns]

                if not available_columns:
                    continue

                df_extracted = df[available_columns]

                # Check if any data was extracted
                if df_extracted.empty:
                    continue

                # Append the extracted data to the compiled DataFrame
                compiled_data = pd.concat([compiled_data, df_extracted], ignore_index=True)

            except Exception as e:
                # Log the error but do not print to reduce console output
                with open(os.path.join(output_csv_path, "errors.log"), 'a') as log_file:
                    log_file.write(f"Error processing {csv_path}: {e}\n")
            
            pbar.update(1)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the compiled DataFrame to the output CSV file
    if not compiled_data.empty:
        compiled_data.to_csv(output_csv_path, index=False)
        print(f"Feature extraction and compilation completed. Output saved to {output_csv_path}")
    else:
        print("No data was extracted from any CSV file. No output file created.")

if __name__ == "__main__":
    # Set the paths to the folders containing input CSV files
    input_folders = [
        "/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/FCT450csv",
        "/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/FCT650csv"
    ]
    # Set the path for the output CSV file
    output_csv_path = "/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/output/compiled_features.csv"

    # Call the function to extract features and compile them into one CSV file
    extract_features_and_compile(input_folders, output_csv_path)
