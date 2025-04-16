import pandas as pd
import glob
import os
import time

def combine_gesture_csv_files(base_path, output_filename, domain_number):
    """
    Combine multiple CSV files containing gesture data into a single DataFrame.
    
    Parameters:
    -----------
    base_path : str
        Path to the directory containing the CSV files.
        Format: "path/to/directory/" (trailing slash required)
    
    output_filename : str
        Filename for the output combined CSV file.
        
    domain_number : int
        Domain number (1 or 4) to specify which dataset type is being processed.
    
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with all gesture data and metadata columns.
    
    Notes:
    ------
    Expected filename format: SubjectX-Y-Z.csv
    Where:
    - X is the subject ID
    - Y is the target/gesture (number for domain 1, figure name for domain 4)
    - Z is the trial ID
    """
    print(f"Starting to combine CSV files from {base_path} (Domain {domain_number})")
    start_time = time.time()
    
    # List to store all DataFrames
    all_dataframes = []
    files_count = 0
    
    # Process all CSV files matching the pattern
    for file_path in glob.glob(base_path + "Subject*.csv"):
        # Extract metadata from filename
        filename = os.path.basename(file_path)
        parts = filename.replace(".csv", "").split("-")
        
        # Extract and clean subject ID, target, and trial ID
        subject_id = parts[0].replace("Subject", "")
        target = parts[1]
        trial_id = parts[2]
        
        try:
            # Read CSV file into DataFrame
            df = pd.read_csv(file_path)
            
            # Add metadata columns
            df["subject_id"] = subject_id
            df["target"] = target
            df["trial_id"] = trial_id
            df["source_file"] = filename
            df["domain"] = domain_number
            
            # Append to list of DataFrames
            all_dataframes.append(df)
            files_count += 1
            
            # Print progress every 100 files
            if files_count % 100 == 0:
                print(f"Processed {files_count} files...")
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
    
    # Combine all DataFrames into one
    print(f"Concatenating {files_count} DataFrames...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save to CSV
    print(f"Saving combined dataset to {output_filename}...")
    combined_df.to_csv(output_filename, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

# Execute the function for both domains
if __name__ == "__main__":
    # Process Domain 1 (numbers 0-9)
    domain1_df = combine_gesture_csv_files(
        base_path="Data/Datasets_CSV/Domain1_csv/",
        output_filename="Data/domain1_dataset.csv",
        domain_number=1
    )
    
    # Process Domain 4 (geometric figures)
    domain4_df = combine_gesture_csv_files(
        base_path="Data/Datasets_CSV/Domain4_csv/",
        output_filename="Data/domain4_dataset.csv",
        domain_number=4
    )
    
    # Display information about Domain 1 dataset
    print("\nDomain 1 Dataset Information:")
    print(f"Number of unique subjects: {domain1_df['subject_id'].nunique()}")
    print(f"Number of unique targets: {domain1_df['target'].nunique()}")
    print(f"Targets: {sorted(domain1_df['target'].unique())}")
    print(f"Number of unique trials: {domain1_df['trial_id'].nunique()}")
    
    # Display information about Domain 4 dataset
    print("\nDomain 4 Dataset Information:")
    print(f"Number of unique subjects: {domain4_df['subject_id'].nunique()}")
    print(f"Number of unique targets: {domain4_df['target'].nunique()}")
    print(f"Targets: {sorted(domain4_df['target'].unique())}")
    print(f"Number of unique trials: {domain4_df['trial_id'].nunique()}")