# Importar las clases necesarias
# Enums
from enums.process_type import ProcessType
from dependency_inyections.container import Container
from preprocessing.time_series_decomposer import TimeSeriesDecomposer
from utils.utils import Utils
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import sys
import os

from training.arima_selector import ARIMASelector
from training.model_selector import ModelSelector


class DataProcessingFacade:
    def __init__(self):
        # Crea una instancia del contenedor y resuelve las dependencias
        self.container = Container()

        self.utils = Utils()

        # Acceder a las instancias de los diferentes componentes
        self.data_explorer = self.container.data_explorer()
        self.model_training = self.container.model_training()
        self.time_series_feature_engineering = self.container.time_series_feature_engineering()
        self.time_series_decomposer = TimeSeriesDecomposer()

    def run(self, type_process):
        print('::: PROCESS: ', type_process)

        if type_process == ProcessType.EXPLORATION_DATA_ANALYSIS.value:
            self.explore_and_analyze_data()
        elif type_process == ProcessType.TRAIN_EVALUATE.value:
            self.process_data_and_save_dataframes()
            self.train_models('data/processed', 'date', 'num_sold')
        elif type_process == ProcessType.TRAIN_AND_EVALUATE_INDIVIDUAL_SERIES.value:
            self.train_models_individual(
                'data/processed', 'df_time_monthly_without_outliers.csv', 'date', 'cost', 'M')
        elif type_process == ProcessType.SELECT_MODEL.value:
            model_selector = ModelSelector()
            weights = {
                'RMSE': 0,
                'MAE': 0,
                'R2': 0,
                'MAPE': 0,
                'SMAPE': 1
            }
            model_selector.load_and_analyze_model_metrics_summary(
                'reports/graficas_modelos/model_metrics.csv', weights)
        elif type_process == ProcessType.FINAL_PREDICT.value:
            pass
            # self.final_predict_process()
        else:
            print("Tipo de proceso desconocido.")

    def explore_and_analyze_data(self):
        data = self.data_explorer.get_dataframe()
        data["dayofyear"] = data.index.dayofyear
        data["month"] = data.index.month
        data["year"] = data.index.year
        dir_seasonal_plot = '/code/reports/decomposition_result/seasonal_plot.png'
        print(data.head(2))
        self.time_series_decomposer.seasonal_plot(
            data, y="cost", period="year", freq="month", output_file=dir_seasonal_plot)

        # data_explorer.plot_time_series(data, date_column='date', target_column='num_sold')
        # data_explorer.calculate_statistics(data, target_column='num_sold')

        # Descomposición de series de tiempo
        # decomposition_result = self.time_series_decomposer.decompose_time_series(data, date_column='date', target_column='num_sold')
        # self.time_series_decomposer.visualize_decomposition(decomposition_result)

        # Otros análisis y exploraciones de datos aquí

    def process_data_and_save_dataframes(self):
        data = self.data_explorer.get_dataframe()
        target_col = 'num_sold'
        group_cols = ['country', 'store', 'product']

        unique_countries = data['country'].unique()
        for country in unique_countries:
            country_data = data[data['country'] == country]
            unique_stores = country_data['store'].unique()

            for store in unique_stores:
                store_data = country_data[country_data['store'] == store]
                unique_products = store_data['product'].unique()

                for product in unique_products:
                    product_data = store_data[store_data['product'] == product]
                    for group, group_data in product_data.groupby(group_cols):
                        self.process_group_data(
                            group_data, target_col, country, store, product)

    def process_group_data(self, group_data, target_col, country, store, product):
        if not isinstance(group_data.index, pd.DatetimeIndex):
            raise ValueError(
                "El índice del DataFrame no es de tipo DatetimeIndex. Asegúrate de que tus fechas estén en el índice.")
        directory = 'data/processed'
        file_name = f"{country}-{store}-{product}.csv"
        self.utils.save_dataframe_as_csv(group_data, directory, file_name)

        # group_data = group_data.sort_index()
        # is_stationary = self.time_series_decomposer.is_stationary(group_data, target_col)
        # group_data_stationary = None

        # if not is_stationary:
        #     group_data_stationary, order_diff_used, new_target_col = self.time_series_decomposer.make_series_stationary(group_data, target_col)
        #     is_stationary = self.time_series_decomposer.is_stationary(group_data_stationary, new_target_col)
        #     group_data['order_diff'] = order_diff_used
        # else:
        #     group_data['order_diff'] = 0

        # print('Stationary=', is_stationary, f"{country} - {store} - {product}")

    def train_models(self, directory, date_col_name, target_col):
        # Obtener una lista de archivos CSV en el directorio
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

        for file_name in csv_files:
            file_path = os.path.join(directory, file_name)
            df = self.utils.load_from_csv(file_path, date_col_name, 'D')
            df = self.add_features_enginiering(df)
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    "El índice del DataFrame no es de tipo DatetimeIndex. Asegúrate de que tus fechas estén en el índice.")
            is_stationary = self.time_series_decomposer.is_stationary(
                df, target_col)
            order_diff_used = 0
            if not is_stationary:
                df_stationary, order_diff_used, new_target_col = self.time_series_decomposer.make_series_stationary(
                    df, target_col)
            # df_to_model = df[target_col]

            # Define diferent models
            # arima_selector = ARIMASelector(
            #     df_to_model, p_range=range(0, 5), q_range=range(0, 5))
            # best_p, best_q = arima_selector.find_optimal_order(order_diff_used)

            # model_order = (best_p, order_diff_used, best_q)  # p, d q
            # self.model_training.set_params_model_arima(model_order)

            # # Define auto arima
            # self.model_training.set_model_auto_arima()

            #
            self.model_training.set_model_xgboost()

            # train an evaluate
            self.model_training.set_datasets(df, target_col)
            self.model_training.train_and_evaluate(target_col)

            # save metrics
            model_metrics = self.model_training.get_model_metrics()

            names_folder = self.model_training.get_names_folder()

            directory = f"/code/reports/graficas_modelos/{names_folder['country']}/{names_folder['store']}/{names_folder['product']}"

            self.utils.save_results_to_file(
                model_metrics, directory, 'model_metrics.csv')
            sys.exit()

    def add_features_enginiering(self, df):
        df['day_of_week'] = df.index.day_of_week
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['year'] = df.index.year
        return df

    def train_models_individual(self, directory, file_name, date_col_name, target_col, freq):
        file_path = os.path.join(directory, file_name)
        df = self.utils.load_from_csv(file_path, date_col_name, freq)

        df, _ = self.time_series_feature_engineering.feature_engineering_time_series_dynamic(
            df)

        # Add fourier features depends on analysis
        # fourier_features = self.time_series_feature_engineering.create_deterministic_matrix(df, 'A',4)
        # df = pd.concat([df, fourier_features], axis=1)

        # print(':: Plot periodogram')
        # self.time_series_feature_engineering.create_deterministic_matrix(df)

        # Set models
        self.model_training.set_model_xgboost()
        self.model_training.set_model_linear_regression()

        # train an evaluate
        self.model_training.set_individual_dataset(df, target_col)
        self.model_training.train_and_evaluate(target_col)
        # save metrics
        model_metrics = self.model_training.get_model_metrics()
        directory = f"/code/reports/graficas_modelos/"
        self.utils.save_results_to_file(
            model_metrics, directory, 'model_metrics.csv')
        sys.exit()
