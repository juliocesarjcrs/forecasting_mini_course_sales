import statsmodels.api as sm
import pandas as pd
# from statsmodels.tsa.statespace import SARIMAXResults

class TimeSeries:

    def __init__(self):
        pass

    def training_model(self, type_model, train, order=(1,1,0), seasonal_order=(0,1,0,12)):
        '''
        Trains a time series model.

        Args:
        - type_model (str): Type of model to train. Currently only supports 'SARIMAX'.
        - train (pd.DataFrame): Dataframe containing the training data with columns 'date' and 'cost'.
        - order (tuple): (p, d, q) order of the ARIMA model. Default is (1,1,0).
        - seasonal_order (tuple): (P, D, Q, S) seasonal order of the ARIMA model. Default is (0,1,0,12).

        Returns:
        - model: The trained time series model.

        Raises:
        - ValueError: If the type of model is not supported.
        '''

        if type_model == 'SARIMAX':
            return sm.tsa.statespace.SARIMAX(train['cost'], exog=train['days_in_month'], order=order, seasonal_order=seasonal_order).fit()
        else:
            raise ValueError('Unsupported model type')

    def predict_model(self, model, params):
        """
        Generates predictions for a given SARIMAX model using the provided test data and exogenous variables.

        Parameters:
        model (SARIMAXResultsWrapper): A trained SARIMAX model.
        params (dict): A dictionary containing the following parameters:
            - train (pandas DataFrame): The training data used to fit the model.
            - test (pandas DataFrame): The test data used to generate predictions.
            - var_exoge (list): A list of column names for the exogenous variables in the test data.

        Returns:
        A pandas Series containing the predicted values for the test data.

        Raises:
        TypeError: If any of the input parameters are of an incorrect type.
        """
        # Validate input parameters
        # if not isinstance(model, SARIMAXResults):
        #     raise TypeError("Model argument must be a SARIMAXResultsWrapper object.")
        if not isinstance(params, dict):
            raise TypeError("Params argument must be a dictionary.")
        if not isinstance(params.get("train"), pd.DataFrame):
            raise TypeError("Train argument must be a pandas DataFrame.")
        if not isinstance(params.get("test"), pd.DataFrame):
            raise TypeError("Test argument must be a pandas DataFrame.")
        if not isinstance(params.get("var_exoge"), list):
            raise TypeError("var_exoge argument must be a list.")

        train = params['train']
        test = params['test']
        var_exoge = params['var_exoge']

        forecast_end = len(train) + len(test) - 1
        exogenous_variables_test = test[var_exoge].values.reshape(-1, 1)
        forecast = model.predict(start=len(train), end=forecast_end, exog=exogenous_variables_test)

        return forecast

def predict_future(model_path, data_path, exog_data, periods=5):
    """
    Predicts future values for a time series using a pre-trained model.

    Parameters:
    model_path (str): Path to the pre-trained model file.
    data_path (str): Path to the time series data file.
    exog_data (list): List of exogenous variables for the future periods.
    periods (int): Number of future periods to predict.

    Returns:
    - A pandas dataframe with the predicted values and the corresponding dates as index.
    """
    # Load the pre-trained model
    best_model = joblib.load(model_path)

    # Load the time series data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Get the last date in the data
    last_date = data.index[-1]

    # Generate dates for the future periods
    future_dates = pd.date_range(start=last_date, periods=periods, freq=data.index.freq)

    # Predict future values
    future_predictions = best_model.forecast(steps=periods, exog=[exog_data]*periods, index=future_dates)

    # Create a dataframe with the predicted values and the corresponding dates as index
    future_data = pd.DataFrame(future_predictions, index=future_dates, columns=['predicted_values'])

    return future_data