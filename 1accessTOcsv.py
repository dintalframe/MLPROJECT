import os
import subprocess
from tqdm import tqdm


def convert_all_access_to_csv_mdbtools(folder_path, csv_output_folder):
    # Get all Access database files in the folder (.accdb only)
    access_files = [f for f in os.listdir(folder_path) if f.endswith('.accdb')]
    print(f"Found {len(access_files)} ACCDB Access files.")

    # Ensure output folder exists
    if not os.path.exists(csv_output_folder):
        os.makedirs(csv_output_folder)

    # Create a log file for failed conversions and processed files
    failed_log_path = os.path.join(csv_output_folder, "failed_conversions.log")
    processed_log_path = os.path.join(csv_output_folder, "processed_files.log")

    with open(failed_log_path, 'w') as log_file:
        log_file.write("Failed Conversions:\n")

    with open(processed_log_path, 'w') as processed_file:
        processed_file.write("Processed Files:\n")

    # Create a progress bar for the overall process
    with tqdm(total=len(access_files), desc="Converting Access Databases", unit="file") as pbar:
        # Iterate over each Access database file and convert to CSV
        for access_file in access_files:
            access_db_path = os.path.join(folder_path, access_file)

            # Extract tables using `mdb-tables`
            tables_command = f"mdb-tables -1 \"{access_db_path}\""
            try:
                tables = subprocess.check_output(tables_command, shell=True).decode().splitlines()
            except subprocess.CalledProcessError as e:
                with open(failed_log_path, 'a') as log_file:
                    log_file.write(f"{access_db_path}: Failed to read tables - {e}\n")
                pbar.update(1)
                continue

            if not tables:
                with open(failed_log_path, 'a') as log_file:
                    log_file.write(f"{access_db_path}: No tables found\n")
                pbar.update(1)
                continue

            # Create a single CSV file per Access database
            csv_path = os.path.join(csv_output_folder, f"{os.path.splitext(access_file)[0]}.csv")
            with open(csv_path, 'w') as csv_file:
                has_data = False
                for table in tables:
                    if not table.strip():
                        continue

                    # Extract table data using `mdb-export` and append to the CSV file
                    export_command = f"mdb-export \"{access_db_path}\" \"{table}\""
                    try:
                        table_data = subprocess.check_output(export_command, shell=True).decode()
                        # Skip the first two rows (header information)
                        table_data_lines = [line for line in table_data.splitlines() if line.strip() and not line.startswith('-- Table:')]
                        
                        csv_file.write("\n".join(table_data_lines) + "\n")
                        has_data = True
                    except subprocess.CalledProcessError as e:
                        with open(failed_log_path, 'a') as log_file:
                            log_file.write(f"{access_db_path} - Table '{table}': Failed to export - {e}\n")

                # Remove the CSV file if no data was written
                if not has_data:
                    os.remove(csv_path)
                    with open(failed_log_path, 'a') as log_file:
                        log_file.write(f"{access_db_path}: No data exported\n")
                else:
                    # Log successful processing of the file
                    with open(processed_log_path, 'a') as processed_file:
                        processed_file.write(f"{access_db_path}: Successfully processed\n")

            pbar.update(1)


if __name__ == "__main__":
    # Set the path to the folder containing Access database files
    folder_path = "/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/FCT450"
    # Set the output folder for CSV files
    csv_output_folder = "/mnt/c/Users/dinta/OneDrive/Desktop/mlproj/FCT450csv"

    # Call the function to convert all Access databases in the folder to CSV files
    convert_all_access_to_csv_mdbtools(folder_path, csv_output_folder)
