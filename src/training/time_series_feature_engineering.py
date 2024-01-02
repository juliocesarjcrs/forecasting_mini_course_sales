import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


class TimeSeriesFeatureEngineering:
    def __init__(self):
        pass

    def feature_engineering_time_series_dynamic(self, time_series_data, remove_na=True):
        """
        Perform dynamic feature engineering on time series, adding columns as applicable.

        Parameters:
        - time_series_data: pandas DataFrame or Series. The original time series with a valid time index.
        - remove_na: bool, optional. Indicates whether to remove rows with missing values after feature engineering.

        Returns:
        - DataFrame: A new DataFrame containing the generated features.
        - date_attributes: list of str. The date-related attributes added to the DataFrame.
        """
        # Make a copy of the original data to avoid modifying the input DataFrame
        data = time_series_data.copy()

        # Convert the index to a DatetimeIndex if needed
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Add date-related features if the index is of type DatetimeIndex
        if isinstance(data.index, pd.DatetimeIndex):
            # Get all available date attributes
            date_attributes = ['year', 'month', 'day', 'day_of_week', 'day_of_year']
            # date_attributes = ['year', 'day_of_week', 'day_of_year']

            # Check if hour-related attributes are present
            if data.index.hour.min() > 0:
                date_attributes += ['hour', 'minute', 'second']

            # Check if week-related attributes are present
            # if hasattr(data.index, 'isocalendar'):
            #     iso_week = data.index.isocalendar().week
            #     data['isoweek'] = iso_week
            #     date_attributes += ['isoweek']

            for attr in date_attributes:
                if hasattr(data.index, attr):
                    data[attr] = getattr(data.index, attr)


        # Remove rows with missing values if necessary
        if remove_na:
            data.dropna(inplace=True)

        return data, date_attributes

    def plot_periodogram(self, ts, detrend='linear', ax=None, output_file= None):
        from scipy.signal import periodogram
        fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
        # fs_years = 1  # Un año
        # fs_days = fs_years * 365.25  # Asumiendo un año promedio de 365.25 días

        # fs = pd.Timedelta(days=fs_days)
        freqencies, spectrum = periodogram(
            ts,
            fs=fs,
            detrend=detrend,
            window="boxcar",
            scaling='spectrum',
        )
        if ax is None:
            _, ax = plt.subplots()
        ax.step(freqencies, spectrum, color="purple")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
        ax.set_xticklabels(
            [
                "Annual (1)",
                "Semiannual (2)",
                "Quarterly (4)",
                "Bimonthly (6)",
                "Monthly (12)",
                "Biweekly (26)",
                "Weekly (52)",
                "Semiweekly (104)",
            ],
            rotation=30,
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_ylabel("Variance")
        ax.set_title("Periodogram")
        if output_file:
            plt.savefig(output_file)
            plt.close()  # Cerrar la figura para liberar memoria
        else:
            plt.tight_layout()
            plt.show()
        return ax

    def create_deterministic_matrix(self, data, freq="M", order=2):
        """
        Create a deterministic matrix with Fourier-based seasonal components.

        Parameters:
        - data: pandas DataFrame or Series. The original time series data with a valid time index.
        - freq: str, optional. The frequency of the seasonality (e.g., 'M' for monthly, 'Q' for quarterly).
        - order: int, optional. The number of Fourier components to include.

        Returns:
        - X: pandas DataFrame. The deterministic matrix with seasonal components.
        """
        fourier = CalendarFourier(freq=freq, order=order)
        dp = DeterministicProcess(
            index=data.index,
            constant=True,
            order=1,
            seasonal=True,
            additional_terms=[fourier],
            drop=True,
        )
        X = dp.in_sample()

        return X