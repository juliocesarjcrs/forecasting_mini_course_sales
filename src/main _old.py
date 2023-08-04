from dependency_inyections.container import Container

from preprocessing.time_series_decomposer import TimeSeriesDecomposer
from utils.utils import Utils
import pandas as pd
import sys


def main():
    print('paso 1')
    # Crea una instancia del contenedor y resuelve las dependencias
    container = Container()
    data_explorer = container.data_explorer()
    utils = Utils()
    # data_explorer.plot_time_series_by_country_store(group_cols=[
    #                                'country', 'store', 'product'], target_col='num_sold', output_dir='/code/reports/graficas_ventas')

    # ------------------------------------------------
    data = data_explorer.get_dataframe()
    time_series_decomposer = TimeSeriesDecomposer()

    target_col = 'num_sold'
    group_cols = ['country', 'store', 'product']
    unique_countries = data['country'].unique()
    i = 0
    for country in unique_countries:

        country_data = data[data['country'] == country]

        unique_stores = country_data['store'].unique()

        for store in unique_stores:
            store_data = country_data[country_data['store'] == store]
            unique_products = store_data['product'].unique()
            for product in unique_products:
                product_data = store_data[store_data['product'] == product]
                for group, group_data in product_data.groupby(group_cols):
                    if not isinstance(group_data.index, pd.DatetimeIndex):
                        raise ValueError(
                            "El índice del DataFrame no es de tipo DatetimeIndex. Asegúrate de que tus fechas estén en el índice.")

                # Ordenar por fecha para asegurarse de que el gráfico esté ordenado cronológicamente
                    group_data = group_data.sort_index()
                    # print(group)
                    is_stationary = time_series_decomposer.is_stationary(
                        group_data, target_col)
                    group_data_stationary = None
                    if not is_stationary:
                        group_data_stationary, order_diff_used, new_target_col = time_series_decomposer.make_series_stationary(
                            group_data, target_col)
                        is_stationary = time_series_decomposer.is_stationary(
                            group_data_stationary, new_target_col)
                        group_data['order_diff'] = order_diff_used
                    # if group_data_stationary is None:
                    #     decomposition_result =time_series_decomposer.decompose_time_series(group_data, target_col)
                    # else:
                    #     decomposition_result =time_series_decomposer.decompose_time_series(group_data_stationary, 'stationary')
                    # time_series_decomposer.visualize_decomposition(decomposition_result, '/code/reports/decomposition_result/decomposition_result.png')
                    directory = 'data/processed'
                    file_name = f"{country}-{store}-{product}.csv"
                    utils.save_dataframe_as_csv(
                        group_data, directory, file_name)
                    print('i = ', i, 'Stationary=', is_stationary,
                          f"{country} - {store} - {product}")
                    sys.exit()
                    model_training = container.model_training()
                    model_training.preprocess_data(directory, file_name)
                    i = i+1


# ///////////////////////////////////////
   # Exploración y análisis de datos
    # data_explorer = DataExplorer()
    # data_explorer.plot_time_series(data, date_column='date', target_column='num_sold')
    # data_explorer.calculate_statistics(data, target_column='num_sold')

    # # Descomposición de series de tiempo
    # time_series_decomposer = TimeSeriesDecomposer()
    # decomposition_result = time_series_decomposer.decompose_time_series(data, date_column='date', target_column='num_sold')
    # time_series_decomposer.visualize_decomposition(decomposition_result)

    # # Preprocesamiento (si es necesario)
    # # ...

    # # Entrenamiento de modelos
    # model_trainer = ModelTraining()
    # model = model_trainer.train_model(data, target_column='num_sold')

    # Realizar predicciones para el próximo año
    # ...
# ///////////////////////////////////////
    # model_training = container.model_training()
    # model_training.preprocess_data()
    # Paso 1: Preparación de datos
    # prepare_data()

    # Paso 2: Entrenamiento de modelos
    # train_models()

    # Paso 3: Selección del mejor modelo
    # best_model = select_best_model()

    # Paso 4: Otras tareas o flujos de trabajo


if __name__ == "__main__":
    main()
