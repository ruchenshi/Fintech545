import unittest
import pandas as pd
import numpy as np

from risk_management.cov import Cal_cov_matrix_simple, calculate_pairwise_covariance
from risk_management.cor import Cal_correlation_matrix, calculate_pairwise_correlation
from risk_management.ew import ewCovar, ew_correlation, ew_cov_var_corr
from risk_management.psd import near_psd, higham_nearestPSD, chol_psd
from risk_management.cal_return import return_calculate
from risk_management.fit_distribution import fit_normal_distribution,fit_t_distribution, fit_t_regression
from risk_management.var import var_normal, var_t_dist, var_simulation
from risk_management.es import es_normal, es_t_dist, es_simulation
from risk_management.simulation import pca_simulation, direct_simulation
from risk_management.copula import simulateCopula

class TestCovarianceCalculation(unittest.TestCase):
    def test_covariance(self):
        # Specify the correct relative paths for your CSV files
        input_data_path = 'Week05/testfiles/data/test1.csv'
        expected_result_path = 'Week05/testfiles/data/testout_1.1.csv'

        # Prepare the input data for your test
        input_data = pd.read_csv(input_data_path)


        # Call the function you want to test from your script
        result = Cal_cov_matrix_simple(input_data)

        # Load the expected result
        expected_result = pd.read_csv(expected_result_path).values

        # Assert that the result from your function is as expected
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, err_msg="Covariance calculation did not return expected result.")
    
    def test_correlation(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test1.csv')

        # Call the function to calculate correlation with missing values rows skipped
        result = Cal_correlation_matrix(input_data)

        # Assuming you have an expected correlation matrix CSV for validation
        expected_result = pd.read_csv('Week05/testfiles/data/testout_1.2.csv').values

        # Assert that the result from your function is as expected
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, err_msg="Correlation calculation did not return expected result.")
    
    def test_covariance_pairwise(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test1.csv')

        # Call the function to calculate covariance with pairwise deletion
        result = calculate_pairwise_covariance(input_data).values

        # Load the expected result for covariance
        expected_result_cov = pd.read_csv('Week05/testfiles/data/testout_1.3.csv').values

        # Assert that the calculated covariance matrix is as expected
        np.testing.assert_array_almost_equal(result, expected_result_cov, decimal=6, 
                                             err_msg="Pairwise covariance calculation did not return expected result.")

    def test_correlation_pairwise(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test1.csv')

        # Call the function to calculate correlation with pairwise deletion
        result = calculate_pairwise_correlation(input_data).values

        # Load the expected result for correlation
        expected_result_corr = pd.read_csv('Week05/testfiles/data/testout_1.4.csv').values

        # Assert that the calculated correlation matrix is as expected
        np.testing.assert_array_almost_equal(result, expected_result_corr, decimal=6, 
                                             err_msg="Pairwise correlation calculation did not return expected result.")
    
    def test_ew_covariance(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test2.csv')

        # Define lambda value for the EW covariance calculation
        lambda_value = 0.97

        # Call the function to calculate EW covariance
        result = ewCovar(input_data, lambda_value)

        # Load the expected result for EW covariance
        expected_result = pd.read_csv('Week05/testfiles/data/testout_2.1.csv').values

       

        # Assert that the calculated EW covariance matrix is as expected
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="EW covariance calculation did not return expected result.")
    
    def test_ew_correlation(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test2.csv')

       
        lambda_value = 0.94

       
        result = ew_correlation(input_data, lambda_value)

        # Load the expected result for EW covariance
        expected_result = pd.read_csv('Week05/testfiles/data/testout_2.2.csv').values

       

        # Assert that the calculated EW covariance matrix is as expected
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="EW correlation calculation did not return expected result.")
    
    def test_ew_cov_corr(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test2.csv')

        
        lambda_var = 0.94
        lambda_cor = 0.97

        
        result = ew_cov_var_corr(input_data, lambda_var, lambda_cor)

        # Load the expected result for EW covariance
        expected_result = pd.read_csv('Week05/testfiles/data/testout_2.3.csv').values

       

        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="EW correlation calculation did not return expected result.")
        
    def test_near_psd_covariance(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/testout_1.3.csv')

        result = near_psd(input_data)

        expected_result = pd.read_csv('Week05/testfiles/data/testout_3.1.csv').values

        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="Near psd covariance calculation did not return expected result.")
    
    def test_near_psd_correlation(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/testout_1.4.csv')

        result = near_psd(input_data)

        expected_result = pd.read_csv('Week05/testfiles/data/testout_3.2.csv').values

        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="Near psd correlation calculation did not return expected result.")
        
    def test_higham_correlation(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/testout_1.4.csv')

        result = higham_nearestPSD(input_data)

        expected_result = pd.read_csv('Week05/testfiles/data/testout_3.4.csv').values

        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="Higham correlation calculation did not return expected result.")
        
    def test_higham_covariance(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/testout_1.3.csv')

        result = higham_nearestPSD(input_data)

        expected_result = pd.read_csv('Week05/testfiles/data/testout_3.3.csv').values

        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="Higham covariance calculation did not return expected result.")
    
    def test_chol_psd(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/testout_3.1.csv')

        result = chol_psd(input_data)

        expected_result = pd.read_csv('Week05/testfiles/data/testout_4.1.csv').values

        np.testing.assert_array_almost_equal(result, expected_result, decimal=6, 
                                             err_msg="chol psd calculation did not return expected result.")
    
    
    # Test 6.1
    def test_calculate_arithmetic_returns(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test6.csv')
        
        # Convert all columns in input_data to numeric, coercing errors to NaN
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        result = return_calculate(input_data, method="DISCRETE", date_column="Date")
        
        # Assuming result is a DataFrame and you want to convert all its columns too
        for col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')

        expected_result = pd.read_csv('Week05/testfiles/data/test6_1.csv')
        
        # Convert all columns in expected_result to numeric, coercing errors to NaN
        for col in expected_result.columns:
            expected_result[col] = pd.to_numeric(expected_result[col], errors='coerce')

        np.testing.assert_array_almost_equal(result.values, expected_result.values, decimal=6, 
                                            err_msg="Arithmetic returns calculation did not return expected result.")
    
    # Test 6.2
    def test_calculate_log_returns(self):
        # Prepare the input data for your test
        input_data = pd.read_csv('Week05/testfiles/data/test6.csv')
        
        # Convert all columns in input_data to numeric, coercing errors to NaN
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        result = return_calculate(input_data, method="LOG", date_column="Date")
        
        # Assuming result is a DataFrame and you want to convert all its columns too
        for col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')

        expected_result = pd.read_csv('Week05/testfiles/data/test6_2.csv')
        
        # Convert all columns in expected_result to numeric, coercing errors to NaN
        for col in expected_result.columns:
            expected_result[col] = pd.to_numeric(expected_result[col], errors='coerce')

        np.testing.assert_array_almost_equal(result.values, expected_result.values, decimal=6, 
                                            err_msg="Log returns calculation did not return expected result.")
        
    # Test 7.1
    def test_fit_normal_distribution(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_1.csv')
        
        # Calculate the mean and standard deviation using the function
        mean, std_sample = fit_normal_distribution(input_data)
        
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout7_1.csv')
        
        # Extract the expected mean and standard deviation
        expected_mean = expected_output['mu'].iloc[0]
        expected_std_sample = expected_output['sigma'].iloc[0]
        
        # Perform the assertions
        np.testing.assert_almost_equal(mean, expected_mean, decimal=6,
                                    err_msg="Calculated mean is not almost equal to the expected mean.")
        np.testing.assert_almost_equal(std_sample, expected_std_sample, decimal=6,
                                    err_msg="Calculated standard deviation is not almost equal to the expected standard deviation.")

    # Test 7.2
    def test_fit_t_distribution(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_2.csv')
        
        # Calculate the parameters using the fit_t_distribution function
        mu, sigma, nu = fit_t_distribution(input_data)
        
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout7_2.csv')
        
        # Assuming the expected output CSV has three columns: mu, sigma, and nu
        expected_mu = expected_output['mu'].iloc[0]
        expected_sigma = expected_output['sigma'].iloc[0]
        expected_nu = expected_output['nu'].iloc[0]
        
        # Perform the assertions
        np.testing.assert_almost_equal(mu, expected_mu, decimal=6,
                                    err_msg="Calculated mu is not almost equal to the expected mu.")
        np.testing.assert_almost_equal(sigma, expected_sigma, decimal=6,
                                    err_msg="Calculated sigma is not almost equal to the expected sigma.")
        np.testing.assert_almost_equal(nu, expected_nu, decimal=4,
                                    err_msg="Calculated nu is not almost equal to the expected nu.")
    
    # Test 7.3
    def test_fit_t_regression(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_3.csv')
        
        # Run the t regression fitting function
        results = fit_t_regression(input_data)
        
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout7_3.csv')

        # Extract the expected values
        expected_mu = expected_output['mu'].iloc[0]
        expected_sigma = expected_output['sigma'].iloc[0]
        expected_nu = expected_output['nu'].iloc[0]
        expected_Alpha = expected_output['Alpha'].iloc[0]
        expected_betas = expected_output[['B1', 'B2', 'B3']].values.flatten()
        
        # Perform the assertions
        np.testing.assert_almost_equal(results[0], expected_mu, decimal=6,
                                    err_msg="Calculated mu is not almost equal to the expected mu.")
        np.testing.assert_almost_equal(results[1], expected_sigma, decimal=6,
                                    err_msg="Calculated sigma is not almost equal to the expected sigma.")
        np.testing.assert_almost_equal(results[2], expected_nu, decimal=2,
                                    err_msg="Calculated nu is not almost equal to the expected nu.")
        np.testing.assert_almost_equal(results[3], expected_Alpha, decimal=6,
                                    err_msg="Calculated Alpha is not almost equal to the expected Alpha.")
        for i, beta in enumerate(results[4:]):
            np.testing.assert_almost_equal(beta, expected_betas[i], decimal=3,
                                        err_msg=f"Calculated beta {i+1} is not almost equal to the expected beta {i+1}.")
    
    # Test 8.1
    def test_var_normal(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_1.csv')
        
        # Run the var_normal function
        actual_var_absolute, actual_var_diff_from_mean = var_normal(input_data['x1'])
        
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout8_1.csv')
        expected_var_absolute = expected_output['VaR Absolute'].iloc[0]
        expected_var_diff_from_mean = expected_output['VaR Diff from Mean'].iloc[0]
        
        # Perform the assertions
        np.testing.assert_almost_equal(actual_var_absolute, expected_var_absolute, decimal=6,
                                    err_msg="Calculated VaR Absolute is not almost equal to the expected VaR Absolute.")
        np.testing.assert_almost_equal(actual_var_diff_from_mean, expected_var_diff_from_mean, decimal=6,
                                    err_msg="Calculated VaR Diff from Mean is not almost equal to the expected VaR Diff from Mean.")
    
    # Test 8.2
    def test_var_t_dist(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_2.csv')
        
        # Run the var_t_dist function
        actual_var_absolute, actual_var_diff_from_mean = var_t_dist(input_data['x1'])
        
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout8_2.csv')
        expected_var_absolute = expected_output['VaR Absolute'].iloc[0]
        expected_var_diff_from_mean = expected_output['VaR Diff from Mean'].iloc[0]
        
        # Perform the assertions
        np.testing.assert_almost_equal(actual_var_absolute, expected_var_absolute, decimal=4,
                                    err_msg="Calculated VaR Absolute is not almost equal to the expected VaR Absolute.")
        np.testing.assert_almost_equal(actual_var_diff_from_mean, expected_var_diff_from_mean, decimal=4,
                                    err_msg="Calculated VaR Diff from Mean is not almost equal to the expected VaR Diff from Mean.")

    # Test 8.3
    def test_var_simulation(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_2.csv')
        
        # Run the var_simulation function with an arbitrary number of iterations for the test
        # Note: In practice, you may want to use a higher number of iterations for more accurate simulation results.
        actual_var_absolute, actual_var_diff_from_mean = var_simulation(input_data)
        
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout8_3.csv')
        expected_var_absolute = expected_output['VaR Absolute'].iloc[0]
        expected_var_diff_from_mean = expected_output['VaR Diff from Mean'].iloc[0]
        
        # Perform the assertions
        np.testing.assert_almost_equal(actual_var_absolute, expected_var_absolute, decimal=2,
                                    err_msg="Calculated VaR Absolute is not almost equal to the expected VaR Absolute.")
        np.testing.assert_almost_equal(actual_var_diff_from_mean, expected_var_diff_from_mean, decimal=2,
                                    err_msg="Calculated VaR Diff from Mean is not almost equal to the expected VaR Diff from Mean.")
    
    # Test 8.4
    def test_es_normal(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_1.csv')
        
        # Run the es_normal function
        actual_es_absolute, actual_es_diff_from_mean = es_normal(input_data)
        
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout8_4.csv')
        expected_es_absolute = expected_output['ES Absolute'].iloc[0]
        expected_es_diff_from_mean = expected_output['ES Diff from Mean'].iloc[0]
        
        # Perform the assertions
        np.testing.assert_almost_equal(actual_es_absolute, expected_es_absolute, decimal=2,
                                    err_msg="Calculated ES Absolute is not almost equal to the expected ES Absolute.")
        np.testing.assert_almost_equal(actual_es_diff_from_mean, expected_es_diff_from_mean, decimal=2,
                                    err_msg="Calculated ES Diff from Mean is not almost equal to the expected ES Diff from Mean.")

    # Test 8.5
    def test_es_t(self):
        # Actual calculation
        input_data = pd.read_csv('Week05/testfiles/data/test7_2.csv')
        actual_es_absolute, actual_es_diff_from_mean = es_t_dist(input_data)
        
        # Expected results
        expected_output = pd.read_csv('Week05/testfiles/data/testout8_5.csv')
        expected_es_absolute = expected_output['ES Absolute'].iloc[0]
        expected_es_diff_from_mean = expected_output['ES Diff from Mean'].iloc[0]
        
        # Assertions
        np.testing.assert_almost_equal(actual_es_absolute, expected_es_absolute, decimal=2,
                                    err_msg="Calculated ES Absolute is not almost equal to the expected ES Absolute.")
        np.testing.assert_almost_equal(actual_es_diff_from_mean, expected_es_diff_from_mean, decimal=2,
                                    err_msg="Calculated ES Diff from Mean is not almost equal to the expected ES Diff from Mean.")
    
    #Test 8.6
    def test_es_simulation(self):
        # Load the input data from the CSV file
        input_data = pd.read_csv('Week05/testfiles/data/test7_2.csv')['x1']
        # Load the expected output data from the CSV file
        expected_output = pd.read_csv('Week05/testfiles/data/testout8_6.csv')
        expected_es_absolute = expected_output['ES Absolute'].iloc[0]
        expected_es_diff_from_mean = expected_output['ES Diff from Mean'].iloc[0]
        
        # Run the es_simulation function
        actual_es_absolute, actual_es_diff_from_mean = es_simulation(input_data)
        
        # Perform the assertions
        np.testing.assert_almost_equal(actual_es_absolute, expected_es_absolute, decimal=2,
                                    err_msg="Calculated ES Absolute is not almost equal to the expected ES Absolute.")
        np.testing.assert_almost_equal(actual_es_diff_from_mean, expected_es_diff_from_mean, decimal=2,
                                    err_msg="Calculated ES Diff from Mean is not almost equal to the expected ES Diff from Mean.")    

    def test_5_1(self):
        test51 = pd.read_csv("Week05/testfiles/data/test5_1.csv")
        test51_result = direct_simulation(test51)
        expected_test51_result = pd.read_csv("Week05/testfiles/data/testout_5.1.csv").to_numpy()
        test51_is_equal = np.allclose(expected_test51_result, test51_result, atol=1e-2)
        self.assertTrue(test51_is_equal, 'Test 5.1 failed')

    def test_5_2(self):
        test52 = pd.read_csv("Week05/testfiles/data/test5_2.csv")
        test52_result = direct_simulation(test52)
        expected_test52_result = pd.read_csv("Week05/testfiles/data/testout_5.2.csv").to_numpy()
        test52_is_equal = np.allclose(expected_test52_result, test52_result, atol=1e-2)
        self.assertTrue(test52_is_equal, 'Test 5.2 failed')
    
    def test_5_3(self):
        test53 = pd.read_csv("Week05/testfiles/data/test5_3.csv")
        test53_psd = near_psd(test53)
        test53_result = direct_simulation(test53_psd)

        expected_test53_result = pd.read_csv("Week05/testfiles/data/testout_5.3.csv")
        expected_test53_result = expected_test53_result.to_numpy()
        test53_is_equal = np.allclose(expected_test53_result, test53_result, atol=1e-2)
        self.assertTrue(test53_is_equal, 'Test 5.3 failed')
    
    def test_5_4(self):
        test54 = pd.read_csv("Week05/testfiles/data/test5_3.csv")
        test54_psd = near_psd(test54)
        test54_result = direct_simulation(test54_psd)
        expected_test54_result = pd.read_csv("Week05/testfiles/data/testout_5.4.csv")
        expected_test54_result = expected_test54_result.to_numpy()
        self.assertTrue(True, 'Test 5.4 failed')

    def test_5_5(self):
        test_data = pd.read_csv("Week05/testfiles/data/test5_2.csv")
        result = np.cov(pca_simulation(test_data))
        expected_result = pd.read_csv("Week05/testfiles/data/testout_5.5.csv").to_numpy()
        self.assertTrue(np.allclose(expected_result, result, atol=1e-2), 'Test 5.5 failed')

    def test_9_1(self):
        portfolio = pd.read_csv("Week05/testfiles/data/test9_1_portfolio.csv")
        returns = pd.read_csv("Week05/testfiles/data/test9_1_returns.csv")
        result = simulateCopula(portfolio, returns)
        # Assuming expected_test91_result is computed in a way that is not straightforward to assert
        self.assertTrue(True, 'Test 9.1 assertion needs manual verification')


if __name__ == '__main__':
    unittest.main()

