#!/usr/bin.env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

#Defaults
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time #benchmarking
import requests #retrievefrom postgREST API

from datetime import datetime
from numpy import mean, std
from scipy.stats import pearsonr, spearmanr

from future import standard_library

#Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer

#Modeling
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

#Feature Selection
from pyHSICLasso import HSICLasso

#Scoring
from sklearn.model_selection import LeaveOneOut, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

#Optimization
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK

#Model Saving
import pickle


standard_library.install_aliases()


#### Accessory Functions
def load_files_by_prefix(dir_path, 
                         word, 
                         file_extension='.tsv', 
                         sep='\t', 
                         use_dask = False, 
                         sample = 256000, 
                         blocksize = 'default', 
                         assume_missing = False):
    """
    Scans the directory for files that start with the given word, loads them into a dictionary 
    with the key being the substring after "word_" and the value being the Dask/pandas DataFrame.
    
    Parameters:
        dir_path (str): Path to the directory to scan.
        word (str): Prefix word to search for in file names.
        file_extension (str): Extension of the files to load (default is '.tsv').
        sep (str): Separator for the file (default is tab).
        use_dask (bool): Set to True for dask output
        
    Returns:
        dict: Dictionary with keys as the part of the filename after 'word_' and values as Dask DataFrames.
    """
    # Start the timer
    start_time = time.time()
    
    # Initialize result dict
    result_dict = {}
    
    # Ensure 'word' is a list to handle both string and list inputs
    if isinstance(word, str):
        word = [word]
    
    # Loop through the files in the directory
    for filename in os.listdir(dir_path):
        if any(w in filename for w in word) and filename.endswith(file_extension):
            print(filename.split(f"{word}_", 1)[-1])
            # Extract the part of the filename after 'word_'
            key = filename.replace(file_extension, "")
            # Build the full file path
            file_path = os.path.join(dir_path, filename)
            if use_dask:
                
                df = dd.read_csv(file_path, 
                                 sep = sep, 
                                 blocksize = blocksize, 
                                 sample = sample, 
                                 assume_missing = False)
            else:
                # Load the file into a Dask DataFrame
                df = pd.read_csv(file_path, sep=sep)
            
            # Store the DataFrame in the dictionary
            result_dict[key] = df
    
    # Stop the timer and calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nElapsed time: {elapsed_time}\n")
    
    return result_dict



def fetch_data_from_postgrest_in_batches(base_url, table, column_name, selected_list, batch_size=50):
    """
    Fetches data from a PostgREST API table in batches, based on a list of selected IDs, preserving the order.

    Parameters:
        base_url (str): The base URL of the PostgREST API.
        table (str): The endpoint or table name to query.
        column_name (str): The column name to filter by.
        selected_list (list): The list of IDs to use for the 'in' filter (preserves order).
        batch_size (int): Number of IDs to include in each batch.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered data from the API, maintaining the original order.
    """
    merged_df = pd.DataFrame()  # Initialize an empty DataFrame for merging batches

    # Process the selected list in batches
    for i in range(0, len(selected_list), batch_size):
        # Create a batch of selected IDs
        selected_batch = selected_list[i:i + batch_size]
        selected_ids_str = ','.join(map(str, selected_batch))  # Ensure IDs are strings

        # Define the query parameter for this batch
        query_params = {
            column_name: f'in.({selected_ids_str})'
        }

        # Make the GET request for this batch
        response = requests.get(f"{base_url}/{table}", params=query_params)

        # Check if the request was successful
        if response.status_code == 200:
            # Load the JSON response into a Pandas DataFrame
            data = response.json()
            df = pd.DataFrame(data)
            
            # Reorder the DataFrame to match the original order of the batch
            df[column_name] = df[column_name].astype(str)
            df = df.set_index(column_name).reindex(selected_batch).reset_index()

            # Append the batch to the merged DataFrame
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        else:
            print(f"Failed to retrieve data for batch starting at index {i}: {response.status_code}")
            continue

    return merged_df



def merge_tsv_files(folder_path):
    """
    Merges tsv files in a given folder_path:
    
    Parameters:
        folder_path (str): string of the folder path
    Returns:
        pd.DataFrame : the merged dataframe
    """
    # Initialize an empty list to store each DataFrame
    df_list = []
    
    # Loop through all the files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a TSV file
        if file_name.endswith('.tsv'):
            file_path = os.path.join(folder_path, file_name)
            # Read the TSV file into a DataFrame
            df = pd.read_csv(file_path, sep='\t')
            # Append the DataFrame to the list
            df_list.append(df)
            
    # Merge all the DataFrames in the list
    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
    else:
        merged_df = pd.DataFrame()  # In case no files are found
    
    return merged_df



def merge_dict_dfs_T(base_dict, 
                     to_merge, 
                     base_key='', 
                     merge_on='', 
                     how='outer'):
    """
    Merges multiple DataFrames from a given dictionary based on a specified key.

    Parameters:
        training_pd (dict): Dictionary containing the DataFrames to merge.
        to_merge (list): List of keys representing the DataFrames to merge.
        base_key (str): Key of the base DataFrame to start merging from (default is 'subject_specimen').
        merge_on (str): Column name to merge on (default is 'specimen_id').
        how (str): Type of merge to perform (default is 'outer'; can also be 'inner').

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Start with the base DataFrame
    merged_df = base_dict[base_key]
    merged_df['specimen_id'] = merged_df['specimen_id'].astype(str)
    
    # Merge each DataFrame specified in to_merge
    for key in to_merge:
        transposed_df = base_dict[key].T.reset_index()
        #Convert number to str
        transposed_df['index'] = transposed_df['index'].astype(str)
        transposed_df.rename(columns={'index': 'specimen_id'}, inplace=True)

        merged_df = pd.merge(merged_df, transposed_df, on=merge_on, how=how)
    
    return merged_df



def rename_columns_based_on_gene_symbols(df, 
                                         gene_df, 
                                         id_col='versioned_ensembl_gene_id', 
                                         symbol_col='gene_symbol'):
    """
    Rename matching columns of the input DataFrame based on the mapping of versioned_ensembl_gene_id to gene_symbol.

    Parameters:
        df (pd.DataFrame): The DataFrame with columns to be renamed.
        gene_df (pd.DataFrame): The DataFrame containing the mapping of versioned_ensembl_gene_id to gene_symbol.
        id_col (str): Column name in gene_df that contains the versioned_ensembl_gene_id. Default is 'versioned_ensembl_gene_id'.
        symbol_col (str): Column name in gene_df that contains the gene_symbol. Default is 'gene_symbol'.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    # Step 1: Create a mapping dictionary from gene_df (only for matching IDs in df.columns)
    id_to_symbol_map = dict(zip(gene_df[id_col].astype(str), gene_df[symbol_col]))

    # Step 2: Filter the mapping to include only the column names present in df
    rename_dict = {col: id_to_symbol_map[col] for col in df.columns.astype(str) if col in id_to_symbol_map}

    # Step 3: Rename the matching columns using the dictionary
    df = df.rename(columns=rename_dict)

    return df, id_to_symbol_map



def long_to_wide(df, id_vars, time_col='timepoint', exclude_id_vars = None):
    """
    Transforms a long DataFrame to wide format by appending the timepoint to column names.

    Parameters:
        df (pd.DataFrame): Input long DataFrame.
        id_vars (list): List of column names to keep as-is.
        time_col (str): Column name for the timepoint (default is 'timepoint').

    Returns:
        pd.DataFrame (wide): Wide DataFrame with updated column names.
        pd.DataFrame (long): Long DataFrame with updated column names.
        
    """
    # Save the identifiers metadata
    metadata = id_vars.copy()
    
    # Reshape the DataFrame
    df_long = df.melt(id_vars=metadata, var_name='Feature', value_name='Value')
    
    # Create a new column that combines the feature name and timepoint
    df_long['Feature_Time'] = df_long['Feature'] + '_D' + df_long[time_col].astype(str)
    
    # Drop the original feature and timepoint columns
    df_long = df_long.drop(columns=['Feature', time_col])
    
    # Check if the timepoint column is in id_vars and remove it
    if time_col in metadata:
        metadata.remove(time_col)
    
    # Pivot the DataFrame back to wide format
    df_wide = df_long.pivot(index=metadata, 
                            columns='Feature_Time', 
                            values='Value').reset_index()
    
    # Ensure no "Feature_Time" column in the index, making them actual column names
    df_wide.columns.name = None
    
    # Group by metadata and Feature_Time to check for duplicates
    duplicate_check = df_long.groupby(metadata + ['Feature_Time']).size().reset_index(name='Count')
    duplicates = duplicate_check[duplicate_check['Count'] > 1]
    
    if not duplicates.empty:
        print(f"Warning: Found {len(duplicates)} duplicates.")
        print(duplicates)
        
    # Remove 'specimen_id' or other exclude_id_vars from the grouping for aggregation
    if exclude_id_vars:
        metadata_for_aggregation = [col for col in metadata if col not in exclude_id_vars]
    else:
        metadata_for_aggregation = metadata
    
    # Group by relevant metadata and aggregate to ensure one row per subject
    # Use 'first' to take the first non-null value for each feature (since there shouldn't be duplicates)
    df_wide = df_wide.groupby(metadata_for_aggregation, as_index=False).first()
    
    # Drop excluded ID vars from the final wide DataFrame if they still exist
    df_wide.drop(columns=[col for col in exclude_id_vars if col in df_wide.columns], inplace=True, errors='ignore')
    
    return df_wide, df_long



def compute_age(df, dob_col, visit_date_col):
    """
    Computes the Age based on the date of birth and visit date.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    dob_col (str): The name of the column containing the date of birth.
    visit_date_col (str): The name of the column containing the visit date.

    Returns:
    pd.DataFrame: The DataFrame with an additional 'Age' column.
    """
    # Ensure that dob_col and visit_date_col are in datetime format
    df[dob_col] = pd.to_datetime(df[dob_col], errors='coerce')
    df[visit_date_col] = pd.to_datetime(df[visit_date_col], errors='coerce')
    
     # Compute age in years
    df['age_at_boost_years'] = df[visit_date_col].dt.year - df[dob_col].dt.year
    df['age_at_boost_years'] -= (df[visit_date_col].dt.month < df[dob_col].dt.month) | (
        (df[visit_date_col].dt.month == df[dob_col].dt.month) & (df[visit_date_col].dt.day < df[dob_col].dt.day)
    )
    
    # Compute age in days and weeks
    df['age_at_boost_days'] = (df[visit_date_col] - df[dob_col]).dt.days
    df['age_at_boost_weeks'] = df['age_at_boost_days'] // 7
    
    return df



def merge_meta_to_df(df1, df2, id_col, dob_col, visit_date_col, as_year = False):
    """
    Merges the age information computed from the second DataFrame into the first DataFrame based on subject_id.
    
    Parameters:
    df1 (pd.DataFrame): The first DataFrame to which age will be added.
    df2 (pd.DataFrame): The second DataFrame containing IDs and age information.
    id_col (str): The name of the column containing the IDs.
    dob_col (str): The name of the column containing the date of birth in the second DataFrame.
    visit_date_col (str): The name of the column containing the visit date in the second DataFrame.

    Returns:
    pd.DataFrame: The merged DataFrame with age information from df2 added to df1.
    """
    # Compute age in the second DataFrame
    df2_with_age = compute_age(df2, dob_col, visit_date_col)
    
    if as_year:
        # Merge the two DataFrames on id_col
        merged_df = df1.merge(df2_with_age[[id_col, 
                                            'age_at_boost_years', 
                                            'ethnicity', 
                                            'race']], 
                              on=id_col, 
                              how='left')
    else:    
        # Merge the two DataFrames on id_col
        merged_df = df1.merge(df2_with_age[[id_col, 
                                            'age_at_boost_years', 
                                            'age_at_boost_weeks', 
                                            'age_at_boost_days', 
                                            'ethnicity', 
                                            'race']], 
                              on=id_col, 
                              how='left')
    
    return merged_df



# Custom functions for correlation metrics on predicted results
def pearson_corr(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def main():
    
    # path to preprocessed data
    dir_processed_data = "./data/harmonized/master_allData_batchCorrected_TSV/"
    
    
    #### Load all input files into a dictionary
    all_pd = load_files_by_prefix(dir_processed_data, word = "tsv", use_dask = False)
    
    # Save list of colnames in each dataset    
    facs_cols = list(all_pd["pbmc_cell_frequency_batchCorrected_data"].T.columns)
    print(f"Len FACS matrix: {len(facs_cols)}")
    gex_cols = list(all_pd["pbmc_gene_expression_tpm_batchCorrected_data"].T.columns)
    print(f"Len GEX matrix: {len(gex_cols)}")
    ab_cols = list(all_pd["plasma_ab_titer_batchCorrected_data"].T.columns)
    print(f"Len AB matrix: {len(ab_cols)}")
    prot_cols = list(all_pd["plasma_cytokine_concentrations_by_olink_batchCorrected_data"].T.columns)
    print(f"Len OLINK matrix: {len(prot_cols)}")
    t_act_cols = list(all_pd["t_cell_activation_raw_data"].T.columns) 
    print(f"Len T-ACT matrix: {len(t_act_cols)}")
    t_pol_cols = list(all_pd["t_cell_polarization_raw_data"].T.columns)    
    print(f"Len T_POL matrix: {len(t_pol_cols)}")
    
    
    
    #### FETCH gene data from the CMI-PB API and create genes reference dataframe
    base_url = 'https://www.cmi-pb.org:443/api/v5'
    table = 'gene'
    column_name = 'versioned_ensembl_gene_id'
    selected_ids = all_pd['pbmc_gene_expression_tpm_batchCorrected_data'].index.tolist()
    
    gene_df = fetch_data_from_postgrest_in_batches(base_url, 
                                                   table, 
                                                   column_name, 
                                                   selected_ids, 
                                                   batch_size = 100)
    
    
    #### MERGE all datasets (after transposing) on specimen_id
    
    # Selected Datasets for merging
    to_merge = [
        "pbmc_cell_frequency_batchCorrected_data",                          #(39, 546)
        "pbmc_gene_expression_tpm_batchCorrected_data",                     #(6603, 666)
        "plasma_ab_titer_batchCorrected_data",                              #(31, 867)
        "plasma_cytokine_concentrations_by_olink_batchCorrected_data",      #(45, 490)
        "t_cell_activation_raw_data",                                       #(3, 299)
        "t_cell_polarization_raw_data"                                      #(6, 261)
    ]
    
    ## Merging
    merged_df = merge_dict_dfs_T(all_pd, to_merge, base_key = "subject_specimen", merge_on = "specimen_id", how = "outer")    
        
    
    #### RENAME columns based on gene symbols and update the values in gex_cols list
    merged_df_gsymbol, mapping_dict = rename_columns_based_on_gene_symbols(merged_df, gene_df)
    gex_cols = [mapping_dict.get(item, item) for item in gex_cols]
    
    #### ADD AGE information
    
    subj_data_folder = "./data/subject_metadata/" #### This folder contains the subjects table for each year
    df_subjects = merge_tsv_files(subj_data_folder)
    
    # Merge age information into the first DataFrame
    merged_data = merge_meta_to_df(merged_df_gsymbol, 
                                  df_subjects, 
                                  "subject_id", 
                                  "year_of_birth", 
                                  "date_of_boost")
    
    #output_path = "./Train/merged_training_matrix_allmeta.csv"
    
    # Save the merged DataFrame to the specified output path
    #merged_data.to_csv(output_path,   
    #                   index=False)
    #print(f"Data successfully saved to {output_path}")

    
    
    #### Split into training and challenge df
    
    training_df = merged_data.loc[merged_data['dataset'] != "2023_dataset"]
    challenge_df = merged_data.loc[merged_data['dataset'] == "2023_dataset"]
    
    
    
    ######################################     TRAINING DATASET PREPARATION
    
    
    #### 1) FILTER subjects without gex data collected
    
    # Get the gene symbols from gene_df
    gene_columns = gene_df["gene_symbol"].values

    # Group by 'subject_id'
    grouped = training_df.groupby("subject_id")

    # List to store subject_ids with all NA in gene columns
    subject_ids_with_all_na = []

    # Iterate through each group (each subject_id)
    for subject_id, group in grouped:
        # Filter the group to only include the gene columns
        group_gene_cols = group[gene_columns]
        
        # Check if all gene columns in the group are NA (for all rows in the group)
        if group_gene_cols.isna().all(axis=None):  # Check across both rows and columns
            # If all gene columns are NA for the entire group, store the subject_id
            subject_ids_with_all_na.append(subject_id)
            
    
    # APPLY FILTER: keep only those subjects with gene expression data
    training_df_filt_ge = training_df[~training_df["subject_id"].isin(subject_ids_with_all_na)]    
    
    #### 2) FILTER: KEEP ONLY subjects with both baseline and target outcomes measured
    
    outcomes = ["subject_id", "dataset", "timepoint", "Monocytes", "CCL3", "IgG_PT"]

    filt_outcomes_cols = training_df_filt_ge[training_df_filt_ge["timepoint"].isin([0, 1, 3, 14])][outcomes]
    
    # Function to check the conditions for each group
    def filter_group_wbase(df):
        # Check if the value of "Monocytes" is present at both timepoints 0 and 1
        monocytes_present = (
            df.loc[df["timepoint"] == 1, "Monocytes"].notna().any() and
            df.loc[df["timepoint"] == 0, "Monocytes"].notna().any()
        )
        
        # Check if the value of "CCL3" is present at both timepoints 0 and 3
        ccl3_present = (
            df.loc[df["timepoint"] == 3, "CCL3"].notna().any() and
            df.loc[df["timepoint"] == 0, "CCL3"].notna().any()
        )
        
        # Check if the value of "IgG_PT" is present at both timepoints 0 and 14
        igg_pt_present = (
            df.loc[df["timepoint"] == 14, "IgG_PT"].notna().any() and
            df.loc[df["timepoint"] == 0, "IgG_PT"].notna().any()
        )
        
        # Return True if all conditions are satisfied
        return monocytes_present and ccl3_present and igg_pt_present

    # Apply the filter using groupby and filter
    filtered_df_outcomes_wbase = filt_outcomes_cols.groupby("subject_id").filter(filter_group_wbase)

    
    train_matrix_woutcm = training_df_filt_ge[training_df_filt_ge["subject_id"].isin(
                                                                            filtered_df_outcomes_wbase["subject_id"].unique()
                                                                                     )]
    
    
    # KEEP METADATA INFO AS IS
    id_vars = train_matrix_woutcm.columns[:7].tolist()  # Keeping first 7 columns (equal in training or challenge)
    id_vars = id_vars + ["age_at_boost_weeks", "race", "ethnicity"]
    
    # CREATE WIDE DF
    train_matrix_woutcm_wide, train_matrix_woutcm_long = long_to_wide(train_matrix_woutcm, 
                                                                      id_vars=id_vars, 
                                                                      time_col='timepoint', 
                                                                      exclude_id_vars = ["specimen_id"]
                                                                     )    
        
    #### 3) FILTER: KEEP ONLY baseline (D0) data and outcomes
    
    # List of metadata columns
    metadata = list(train_matrix_woutcm_wide.columns[:8])

    # Filter columns that end with "D0" or are in the exceptions list
    filtered_columns = [
        col for col in train_matrix_woutcm_wide.columns 
        if col.endswith("D0") or col in metadata
    ]
    
    # Create a new DataFrame with the filtered columns (the outcomes are excluded)
    filtered_train_matrix_woutcm_wide = train_matrix_woutcm_wide[filtered_columns]
    
    # Create a DataFrame with the outcomes columns
    outcomes_cols = train_matrix_woutcm_wide[["Monocytes_D1", "CCL3_D3", "IgG_PT_D14"]]
    
    filtered_train_matrix_woutcm_wide = pd.concat([filtered_train_matrix_woutcm_wide, outcomes_cols], axis=1)
    
    
    
    ########################             MODELLING
    
    data = filtered_train_matrix_woutcm_wide.copy()
    
    # List of target outcomes
    output_list = ["Monocytes_D1",
                   "Monocytes_FC", 
                   "CCL3_D3", 
                   "CCL3_FC", 
                   "IgG_PT_D14", 
                   "IgG_PT_FC"]  # Define output columns
    
    
    # Update name of columns by adding "_D0" to each element and appending the outcome
    facs_cols = [col + "_D0" for col in facs_cols] + ["Monocytes_D1"]
    ab_cols = [col + "_D0" for col in ab_cols] + ["IgG_PT_D14"]
    gex_cols = [col + "_D0" for col in gex_cols] + ["CCL3_D3"]
    prot_cols = [col + "_D0" for col in prot_cols]
    t_act_cols = [col + "_D0" for col in t_act_cols]
    t_pol_cols = [col + "_D0" for col in t_pol_cols]
    
    covars_names = ["dataset" ,
                    "infancy_vac", 
                    "biological_sex", 
                    "age_at_boost_weeks", 
                    "ethnicity", 
                    "race"] 
    
    encode_cov = ["infancy_vac", 
                  "biological_sex", 
                  "ethnicity", 
                  "race"]
           
        
    #### 1) Apply log2 transformation directly to specified columns in the dataframe
    data[gex_cols] = np.log2(data[gex_cols] + 1)
    
    #### 2) Apply log2 transformation with offset to specified columns in the dataframe
    #data[ab_cols] = np.log2(data[ab_cols] + 1)
    offset = abs(data[ab_cols].min().min()) + 1
    data[ab_cols] = np.log2(data[ab_cols] + offset)
    
    #### 3) Compute fold change as log2(b/a +1) (before transformation) or (b-a) (on log2 transformed data)
    data["Monocytes_FC"] = np.log2(data["Monocytes_D1"] / data["Monocytes_D0"] + 1)
    data["CCL3_FC"] = data["CCL3_D3"] - data["CCL3_D0"]
    data["IgG_PT_FC"] = data["IgG_PT_D14"] - data["IgG_PT_D0"]
    
    
    #### 4) Apply label encoding or binary encoding to specified columns
    label_encoders = {}
    binary_mappings = {}  # To store mappings for binary-encoded columns

    for col in encode_cov:
        unique_values = data[col].unique()
    
        # Check if the column has only two unique values for binary encoding
        if len(unique_values) == 2:
            # Map the values to 0 and 1 based on the unique values
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            binary_mappings[col] = mapping  # Save the mapping for reversal
            data[col] = data[col].map(mapping)
        else:
            # Apply LabelEncoder for columns with more than two categories
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le  # Store each encoder for potential inverse transform

        # Convert to float if needed
        data[col] = data[col].astype(float)
    
    
    #### 5) kNN IMPUTATION OF MISSING VALUES 
    
    # Initialize the KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)
    
    # Function to process each omics matrix: imputation
    def process_omics(data, columns, imputer):
        data_imputed = imputer.fit_transform(data[columns])
        return pd.DataFrame(data_imputed, columns=columns, index=data.index)
    
    
    
    # Impute each individual omics matrix
    clin_cols = ["infancy_vac", "biological_sex", "age_at_boost_weeks", "ethnicity", "race"]
    data_clin = data[clin_cols]
    data_clin_cp = data_clin.copy()  # Copy to keep original data intact
    
    data_facs = process_omics(data, facs_cols, knn_imputer)
    data_ab = process_omics(data, ab_cols, knn_imputer)
    data_prot = process_omics(data, prot_cols, knn_imputer)
    data_t_act = process_omics(data, t_act_cols, knn_imputer)
    data_t_pol = process_omics(data, t_pol_cols, knn_imputer)
    data_gex = process_omics(data, gex_cols, knn_imputer)  # Assuming gex_cols is defined
    
    data_fc = data[["Monocytes_FC", "CCL3_FC", "IgG_PT_FC"]]
    
    # Recombine the imputed data
    data_processed = pd.concat([data_clin_cp, 
                                data_facs, 
                                data_ab, 
                                data_prot, 
                                data_t_act, 
                                data_t_pol, 
                                data_gex,
                                data_fc], axis=1)
    
    #### 1) SCALE THE IMPUTED DATA FOR HSIC    
    
    # Step 1: Define features (X) by dropping target and non-numeric columns
    X_num = data_processed.drop(columns=["infancy_vac", "biological_sex", "ethnicity", "race"] + output_list)
    
    # Step 2: Extract target columns specified in output_list
    y = data_processed[output_list]
    
    # Step 3: Scale numerical features (X) and targets (y)    
    # Initialize Scaler (yeo-johnson)
    scaler = PowerTransformer()
    
    X_scaled = scaler.fit_transform(X_num)
    X_scaled = pd.DataFrame(X_scaled, columns=X_num.columns, index=X_num.index)

    y_scaled = scaler.fit_transform(y)
    y_scaled = pd.DataFrame(y_scaled, columns=y.columns, index=y.index)
    
    # Step 4: Recombine scaled features, targets, and categorical columns    
    # Ensure that encoded categorical columns (e.g., 'infancy_vac', etc.) are included
    categorical_data = data_clin_cp[["infancy_vac", "biological_sex", "ethnicity", "race"]]  # Only keep necessary columns
    
    data_processed_scaled = pd.concat([categorical_data, X_scaled, y_scaled], axis=1)
    
    # Save to CSV (this will be used as input for HSIC LASSO)
    out_file = "./HSIC_input_matrix.csv"
    data_processed_scaled.to_csv(out_file, index=False) #occhio qui
    
    #### FEATURE SELECTION WITH HSIC LASSO
    
    # Initialize HSICLasso
    hsic_lasso = HSICLasso()
    
    # Set the input for HSIC LASSO using the saved scaled matrix and the list of target outcomes
    hsic_lasso.input(out_file, output_list=output_list)
         
    # Retrieve feature names after calling input to get self.X_in from the loaded data
    feature_columns = [col for col in pd.read_csv(out_file).columns if col not in output_list]

    # Extract the indices of covariates based on covariate names
    covars_index = [feature_columns.index(name) for name in covars_names if name in feature_columns]
    
    # HSIC lasso regression
    hsic_lasso.regression(25, B = 0, covars = hsic_lasso.X_in[covars_index].T, n_jobs = -1)
    
    hsic_lasso.dump()
    hsic_lasso.plot_path()
    
    # Compute linkage and save some plots
    hsic_lasso.linkage()
    hsic_lasso.plot_heatmap()
    hsic_lasso.save_param()
    
    # Get selected features from hsic_lasso
    hsic_features = hsic_lasso.get_features()
    
    # Baseline features to add to the selected features (if not originally selected by HSIC lasso)    
    baseline_feats = ["Monocytes_D0", "CCL3_D0", "IgG_PT_D0", "infancy_vac", 
                      "biological_sex", "age_at_boost_weeks", "ethnicity", "race"]
    
    # Combine hsic features with unique elements from baseline_feats to create HSIC plus
    hsic_plus = hsic_features + [feature for feature in baseline_feats if feature not in hsic_features]
    print(hsic_plus)
    
    # Load unscaled data after feature selection as train dataset
    X = data_processed[hsic_plus]  #Selected features
    y = data_processed[output_list]  # Target columns
    
    # Ensure output directory for plots exists
    output_dir = "prediction_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    
    #### 3) MODEL HYPERPARAMETERS OPTIMIZATION WITH HYPEROPT
    
    def optimize_hyperparam(X, y):
        """
        Optimize hyperparameters for MultiOutputRegressor with CatBoostRegressor using Hyperopt.
        
        Parameters:
            X (pd.DataFrame): Feature matrix.
            y (pd.DataFrame): Multi-output target matrix.
            
        Returns:
            dict: Best hyperparameters found by Hyperopt.
        """
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    
        def objective(params):
            # Initialize MultiOutputRegressor with CatBoostRegressor and given hyperparameters
            model = MultiOutputRegressor(
                CatBoostRegressor(
                    **params,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    early_stopping_rounds=50,
                    random_seed=42,
                    verbose=0
                )
            )
            # Fit on training data and validate on validation data
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            
            # Calculate mean RMSE across all outputs
            rmse = root_mean_squared_error(y_valid, preds)
            return {'loss': rmse, 'status': STATUS_OK}

        # Define the search space for CatBoost hyperparameters
        search_space = {
            'learning_rate': hp.uniform('learning_rate', 0.05, 0.5),
            'iterations': hp.choice('iterations', [100, 300, 500, 700, 1000]),
            'depth': hp.choice('depth', [3, 4, 5, 6, 7, 10]),
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
            'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli'])
        }
    
        # Run Hyperopt optimization
        trials = Trials()
        best_params = fmin(fn=objective, 
                           space=search_space, 
                           algo=tpe.suggest, 
                           max_evals=20, 
                           trials=trials, 
                           rstate=np.random.default_rng(42)
                          )
        
        # Convert categorical choices to actual values
        best_params['iterations'] = [100, 300, 500, 700, 1000][best_params['iterations']]
        best_params['depth'] = [3, 4, 5, 6, 7, 10][best_params['depth']]
        best_params['bootstrap_type'] = ['Bayesian', 'Bernoulli'][best_params['bootstrap_type']]
        
        return best_params
    
    #### 4) MODEL TRAINING AND EVALUATION METRICS
    
    def train_and_evaluate(X, y):
        # Find optimal parameters for MultiOutputRegressor with CatBoost
        best_params = optimize_hyperparam(X, y)
        print("Best parameters found:", best_params)
        
        # Initialize MultiOutputRegressor with tuned CatBoostRegressor
        model = MultiOutputRegressor(
            CatBoostRegressor(
                **best_params,
                loss_function='RMSE',
                eval_metric='RMSE',
                early_stopping_rounds=50,
                random_seed=42,
                verbose=0
            )
        )
        # evaluate model in LOO crossvalidation
        scores = cross_val_score(model, 
                                 X, 
                                 y, 
                                 scoring='neg_root_mean_squared_error', 
                                 cv=LeaveOneOut(), 
                                 n_jobs=-1)
        # force positive
        scores = -scores
        
        # report performance
        print('RMSE: %.3f (%.3f)' % (mean(scores), std(scores)))
        
        # Perform Leave-One-Out Cross-Validation
        all_true, all_preds = [], []
        
        for train_index, test_index in LeaveOneOut().split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Fit the model and make predictions
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            all_preds.append(preds.flatten())
            all_true.append(y_test.values.flatten())
    
        # Ensure all_preds and all_true are structured properly
        all_preds = np.array(all_preds)  # Shape: (n_samples, n_targets)
        all_true = np.array(all_true)    # Shape: (n_samples, n_targets)


        # Create a DataFrame for predictions and true values
        predictions_df = pd.DataFrame(all_preds, columns=[f"Prediction_{name}" for name in y.columns])
        true_values_df = pd.DataFrame(all_true, columns=[f"True_{name}" for name in y.columns])

        # Combine predictions and true values into a single DataFrame
        result_df = pd.concat([predictions_df, true_values_df], axis=1)
        
        # Save to CSV
        result_df.to_csv("MOCBoost_predictions_vs_true_values.csv", index=False)
        print("Predictions and true values saved to 'MOCBoost_predictions_vs_true_values.csv'")
        
        # Initialize a results dictionary to store correlations for each target
        results = {}

        # Iterate through each target (column) to compute correlations
        for i, target in enumerate(y.columns):
            true_values = all_true[:, i]
            predicted_values = all_preds[:, i]
        
            # Calculate ranked Spearman correlation for the target
            spearman_corr_value = spearman_corr(
                pd.Series(true_values).rank().values, pd.Series(predicted_values).rank().values
            )

            # Store the result
            results[f"{target}_ranked_spearman_cor"] = spearman_corr_value

            # Print the result
            print(f"Ranked Spearman Correlation for {target}: {spearman_corr_value:.4f}")

        # Save results to a CSV 
        results_df = pd.DataFrame(list(results.items()), columns=["Target", "Ranked Spearman Correlation"])
        results_df.to_csv("ranked_spearman_correlations.csv", index=False)
        print("Ranked Spearman correlations saved to 'ranked_spearman_correlations.csv'")
        
        # Create a plot for each target
        num_targets = len(y.columns)
        fig, axes = plt.subplots(nrows=num_targets, ncols=1, figsize=(6, 4 * num_targets))
        
        # Ensure axes is iterable if there's only one target
        if num_targets == 1:
            axes = [axes]  
        
        # Scatter plot of true vs pred for each individual target
        for i, target in enumerate(y.columns):
            ax = axes[i]
            true_values = all_true[:, i]
            predicted_values = all_preds[:, i]
        
            # Scatter plot
            ax.scatter(true_values, 
                       predicted_values, 
                       alpha=0.6, 
                       edgecolors="k", 
                       label='Predicted')
            ax.plot([min(true_values), max(true_values)], 
                    [min(true_values), max(true_values)], 
                    color='red', 
                    lw=2, 
                    label='Ideal')
            ax.set_title(f"True vs Predicted for {target}")
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predicted Values")
            ax.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the multi-plot figure
        plt.savefig(f"{output_dir}/true_vs_predicted_per_target.png")
        plt.close()
        print(f"Individual scatter plots per target saved to '{output_dir}/true_vs_predicted_per_target.png'")
        
        # Flatten results to plot overall model performance
        all_preds = all_preds.flatten()  # Shape: (n_samples, n_targets)
        all_true = all_true.flatten()   # Shape: (n_samples, n_targets)
        
        
        # Initialize a results dictionary to store correlations for each target
        results = {}
        results = {
            'pearson.cor.pred.true': pearson_corr(all_true, all_preds),
            'spearman.cor.pred.true': spearman_corr(all_true, all_preds),
            'ranked.spearman.cor.pred.true': spearman_corr(pd.Series(all_true).rank().values,
                                                           pd.Series(all_preds).rank().values),
            'mse': mean_squared_error(all_true, all_preds),
            'mae': mean_absolute_error(all_true, all_preds),
            'rmse': root_mean_squared_error(all_true, all_preds),
            'r2': r2_score(all_true, all_preds)
        }
        
        # Save multi-output plot
        plt.figure(figsize=(6, 6))
        plt.scatter(all_true, 
                    all_preds, 
                    alpha=0.6, 
                    edgecolors="k", 
                    label='Predicted')
        plt.plot([min(all_true), max(all_true)], 
                 [min(all_true), max(all_true)], 
                 color='red', 
                 lw=2, 
                 label='Ideal')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"MOCBoost")
        plt.legend()
        plt.savefig(f"{output_dir}/optimized_regressorChain_output_true_vs_pred.png")
        plt.close()
        
        # Print and return results
        print("Evaluation Results:", results)
        
        #### TRAIN final model on entire Training data
        
        # Initialize MultiOutputRegressor with tuned CatBoostRegressor
        model = MultiOutputRegressor(
            CatBoostRegressor(
                **best_params,
                loss_function='RMSE',
                eval_metric='RMSE',
                early_stopping_rounds=50,
                random_seed=42,
                verbose=0
            )
        )
        
        # Fit model
        model.fit(X, y)
        
        # Save model
        model_path = f"{output_dir}/Optimized_MOCBoost_model.pkl"
        
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        print(f"Final MultiOutputRegressor with CatBoostRegressor saved to '{model_path}'")
        
        
        return results, model    
    
    
    #### Train and evaluate model
    results, model = train_and_evaluate(X, y)
    
    
    #### 5) PREPARE VALIDATION DATA
    
    def preprocess_validation_data(validation_data, 
                                   training_data, 
                                   clin_cols, 
                                   facs_cols, 
                                   ab_cols, 
                                   prot_cols, 
                                   t_act_cols, 
                                   t_pol_cols, 
                                   gex_cols, 
                                   encode_col,  
                                   offset):
        """
        Preprocess validation data to match the transformations applied to training data.
    
        Parameters:
        - validation_data: DataFrame containing raw validation data.
        - training_data: DataFrame containing preprocessed training data (used for reference).
        - clin_cols: List of columns corresponding to clinical (meta) features.
        - facs_cols: List of columns corresponding to FACS features. 
        - ab_cols: List of columns corresponding to antibody features.
        - prot_cols: List of columns corresponding to olink features.
        - t_act_cols: List of columns corresponding to T cell activation features.
        - t_pol_cols: List of columns corresponding to T cell polarization features.
        - gex_cols: List of columns corresponding to gene expression features.
        - encode_cols: List of categorical columns to encode.
        - offset: Offset applied to `ab_cols` during training for log transformation.
    
        Returns:
        - validation_data_processed: Preprocessed validation data ready for model prediction.
        """
        validation_data = validation_data.copy()
        
        # Apply log2 transformation for gene expression features
        validation_data[gex_cols] = np.log2(validation_data[gex_cols] + 1)
        
        # Apply offset and log2 transformation for antibody features
        validation_data[ab_cols] = np.log2(validation_data[ab_cols] + offset)
        
        # Apply label encoding or binary encoding to specified columns
        label_encoders = {}
        for col in encode_col:
            unique_values = validation_data[col].unique()
    
            # Check if the column has only two unique values for binary encoding
            if len(unique_values) == 2:
                # Map the values to 0 and 1 based on the unique values
                validation_data[col] = validation_data[col].map({unique_values[0]: 0, unique_values[1]: 1})
            else:
                # Apply LabelEncoder for columns with more than two categories
                le = LabelEncoder()
                validation_data[col] = le.fit_transform(validation_data[col].astype(str))
                label_encoders[col] = le  # Store each encoder for potential inverse transform
        
            # Convert to float if needed
            validation_data[col] = data[col].astype(float)
        
        # Initialize the KNNImputer
        knn_imputer = KNNImputer(n_neighbors=5)
    
        # Function to process each omics matrix: impute
        def process_omics(data, columns, imputer):
            data_imputed = imputer.fit_transform(data[columns])
            return pd.DataFrame(data_imputed, columns=columns, index=data.index)
        
        # Impute each omics individually
        val_data_facs = process_omics(validation_data, facs_cols, knn_imputer)
        val_data_ab = process_omics(validation_data, ab_cols, knn_imputer)
        val_data_prot = process_omics(validation_data, prot_cols, knn_imputer)
        val_data_t_act = process_omics(validation_data, t_act_cols, knn_imputer)
        val_data_t_pol = process_omics(validation_data, t_pol_cols, knn_imputer)
        val_data_gex = process_omics(validation_data, gex_cols, knn_imputer)  
        
        # Concatenate imputed matrixes
        validation_processed = pd.concat([validation_data[clin_cols], 
                                          val_data_facs,
                                          val_data_ab, 
                                          val_data_prot, 
                                          val_data_t_act, 
                                          val_data_t_pol, 
                                          val_data_gex,
                                         ], axis = 1)
        
        # Ensure alignment with training features
        training_features = training_data.columns  # Extract the feature columns from the training data
        validation_processed = validation_processed[training_features]  # Reorder and drop extra columns
        
        # Create validation data processed dataframe
        validation_data_processed = pd.DataFrame(
            validation_processed, columns=training_features, index=validation_data.index
        )
        
        return validation_data_processed
    

    
    #### Step 1: Keep only baseline D0
    challenge_df_d0 = challenge_df[challenge_df["timepoint"] == 0]
    
    #### Step 2: Convert to wide
    id_vars = challenge_df_d0.columns[:7].tolist()  # Keeping first 7 columns (equal in training or challenge)
    id_vars = id_vars + ["age_at_boost_weeks", "race", "ethnicity"]
    
    challenge_df_d0_wide, challenge_df_d0_long = long_to_wide(challenge_df_d0, 
                                                              id_vars=id_vars, 
                                                              time_col='timepoint', 
                                                              exclude_id_vars = ["specimen_id"])
    
    
    #### PREDICT VALIDATION OUTCOMES
    
    validation_data = challenge_df_d0_wide
    
    # Remove outcomes from the list of columns names for each matrix
    ab_cols_val = ab_cols
    ab_cols_val.remove("IgG_PT_D14")
    
    gex_cols_val = gex_cols
    gex_cols_val.remove("CCL3_D3")
    
    facs_cols_val = facs_cols
    facs_cols_val.remove("Monocytes_D1")
       
    
    # Process validation data
    validation_data_processed = preprocess_validation_data(
        validation_data=validation_data,
        training_data=X,
        clin_cols = clin_cols,
        facs_cols = facs_cols_val,
        ab_cols=ab_cols_val,
        prot_cols = prot_cols,
        t_act_cols = t_act_cols,
        t_pol_cols = t_pol_cols,
        gex_cols=gex_cols_val,
        encode_col = encode_cov,
        offset=offset  # Offset computed from training data
    )
    
    # Use the processed validation data for prediction
    predictions = model.predict(validation_data_processed)
    
    # Save subject_ids values
    subject_ids = validation_data["subject_id"].values
    
    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame(predictions, columns=output_list)
    predictions_df.insert(0, "subject_id", subject_ids)
    
    # Ensure output directory for plots exists
    output_dir = "predictions_results_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions to a CSV file
    predictions_csv_path = os.path.join(output_dir, "2023_predictions_values.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"Predictions saved to {predictions_csv_path}")

    # Create a DataFrame for ranks
    ranks_df = predictions_df.copy()
    for task in output_list:
        ranks_df[task] = ranks_df[task].rank(method="average", ascending=False)

    # Save ranks to a CSV file
    ranks_csv_path = os.path.join(output_dir, "2023_predictions_ranks.csv")
    ranks_df.to_csv(ranks_csv_path, index=False)
    print(f"Ranks saved to {ranks_csv_path}")
    
    
if __name__ == "__main__":
    main()