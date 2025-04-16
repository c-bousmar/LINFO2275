import pandas as pd
import glob
import os
import time

def get_dataset_from_domain(dataset_path, domain_number):
    """
    Load and filter dataset by domain, with appropriate target type casting.

    This function reads the dataset from the specified path, filters it based on
    the provided domain number, and ensures the 'target' column is correctly
    cast to the expected data type depending on the domain:
        - Domain 1: targets are integers
        - Domain 4: targets are strings

    Parameters
    ----------
    dataset_path : str
        Path to the CSV dataset file.
    domain_number : int
        Domain identifier to filter the dataset (1 or 4 supported).

    Returns
    -------
    pandas.DataFrame or None
        Filtered DataFrame with correct 'target' column types if domain is valid;
        otherwise, returns None and prints an error.

    Notes
    -----
    - Only domain 1 and domain 4 are currently supported.
    - If an unsupported domain is provided, the function will return None.
    """
    df = pd.read_csv(dataset_path)
    df = df[df['domain'] == domain_number]
    if (domain_number == 1):
        df['target'] = df['target'].astype(int)
    elif (domain_number == 4):
        df['target'] = df['target'].astype(str)
    else:
        print("Error - Only Domain 1 and 4 Available for now.")
        return None
    return df

def combine_gesture_csv_files(output_filename="Data/dataset.csv"):
    """
    Combine multiple CSV files from multiple domains into a single DataFrame.
    
    Parameters:
    -----------
    output_filename : str
        Filename for the output combined CSV file.
    
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with all gesture data and metadata columns from all domains.
    
    Notes:
    ------
    Expected filename format: SubjectX-Y-Z.csv
    Where:
    - X is the subject ID
    - Y is the target/gesture
    - Z is the trial ID
    """
    print(f"Starting to combine CSV files into a single dataset")
    start_time = time.time()
    
    # List to store all DataFrames
    all_dataframes = []
    total_files_count = 0
    
    # Define domains to process
    domains = {
        1: "Data/Datasets_CSV/Domain1_csv/",  # Numbers 0-9
        4: "Data/Datasets_CSV/Domain4_csv/"   # Geometric figures
    }
    
    # Process each domain
    for domain_number, base_path in domains.items():
        print(f"\nProcessing Domain {domain_number} from {base_path}")
        domain_files_count = 0
        
        # Process all CSV files in this domain
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
                domain_files_count += 1
                total_files_count += 1
                
                # Print progress every 100 files
                if domain_files_count % 100 == 0:
                    print(f"Processed {domain_files_count} files in Domain {domain_number}...")
                    
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
        
        print(f"Finished Domain {domain_number}: {domain_files_count} files processed")
    
    # Combine all DataFrames into one
    print(f"\nConcatenating {total_files_count} DataFrames from all domains...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save to CSV
    print(f"Saving combined dataset to {output_filename}...")
    combined_df.to_csv(output_filename, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

# Execute the function to create a single dataset
if __name__ == "__main__":
    # Combine all domains into a single dataset
    combined_df = combine_gesture_csv_files(output_filename="Data/dataset.csv")
    
    # Display information about the combined dataset
    print("\nCombined Dataset Information:")
    print(f"Number of unique subjects: {combined_df['subject_id'].nunique()}")
    print(f"Number of unique domains: {combined_df['domain'].nunique()}")
    
    # Show information for each domain
    for domain in sorted(combined_df['domain'].unique()):
        domain_df = combined_df[combined_df['domain'] == domain]
        print(f"\nDomain {domain} Information:")
        print(f"  Records: {len(domain_df)}")
        print(f"  Unique subjects: {domain_df['subject_id'].nunique()}")
        print(f"  Unique targets: {domain_df['target'].nunique()}")
        print(f"  Targets: {sorted(domain_df['target'].unique())}")
        print(f"  Unique trials: {domain_df['trial_id'].nunique()}")