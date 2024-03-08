import numpy as np
import pandas as pd

def Cal_cov_matrix_simple(df):
    # Drop rows with any missing values to ensure calculations only use complete cases
    df_clean = df.dropna()
    # Calculate the covariance matrix using the cleaned DataFrame
    cov_mat = df_clean.cov().values
    return cov_mat



def calculate_pairwise_covariance(df):
    """
    Calculates the pairwise covariance matrix of a dataframe.
    
    :param df: A pandas DataFrame with potential missing values
    :return: A pandas DataFrame representing the covariance matrix
    """
    # The .cov() function automatically handles missing data on a pairwise basis
    covariance_matrix = df.cov()
    return covariance_matrix
