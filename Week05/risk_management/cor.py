import pandas as pd

def Cal_correlation_matrix(df):
    # Drop rows with any missing values to ensure calculations only use complete cases
    df_clean = df.dropna()
    # Calculate the correlation matrix using the cleaned DataFrame
    corr_mat = df_clean.corr().values
    return corr_mat

def calculate_pairwise_correlation(df):
    """
    Calculates the pairwise correlation matrix of a dataframe.
    
    :param df: A pandas DataFrame with potential missing values
    :return: A pandas DataFrame representing the correlation matrix
    """
    # The .corr() function automatically handles missing data on a pairwise basis
    correlation_matrix = df.corr()
    return correlation_matrix