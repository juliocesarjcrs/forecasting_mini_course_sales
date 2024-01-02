import pandas as pd
import ast

class ModelSelector:

    def load_and_analyze_model_metrics_summary(self, file_path, weights):
        # Cargar los datos del archivo CSV
        results_df = pd.read_csv(file_path)

        # Convertir la representación de cadena del diccionario en un objeto de diccionario
        results_df['metrics'] = results_df['metrics'].apply(ast.literal_eval)

        # Calcular las métricas adicionales
        results_df['mean_RMSE'] = results_df['metrics'].apply(
            lambda x: x['RMSE'])
        results_df['mean_MAE'] = results_df['metrics'].apply(
            lambda x: x['MAE'])
        results_df['mean_R2'] = results_df['metrics'].apply(lambda x: x['R2'])
        results_df['mean_MAPE'] = results_df['metrics'].apply(
            lambda x: x['MAPE'])
        results_df['mean_SMAPE'] = results_df['metrics'].apply(
            lambda x: x['SMAPE'])

        # Aplicar ponderaciones a las métricas
        results_df['weighted_metric'] = (
            weights['RMSE'] * results_df['mean_RMSE'] +
            weights['MAE'] * results_df['mean_MAE'] +
            weights['R2'] * results_df['mean_R2'] +
            weights['MAPE'] * results_df['mean_MAPE'] +
            weights['SMAPE'] * results_df['mean_SMAPE']
        )

        # Agrupar por 'model_name' y 'dataset' y calcular el promedio de las métricas ponderadas
        summary_df = results_df.groupby(['model_name']).agg(
            {'weighted_metric': 'mean'}).reset_index()

        # Ordenar el dataframe por la métrica ponderada en orden ascendente
        summary_df = summary_df.sort_values('weighted_metric', ascending=True)

        # Mostrar el dataframe con las métricas
        print(summary_df)
