from dependency_injector import containers, providers
from utils.utils import Utils
from training.model_training import ModelTraining
from preprocessing.data_explorer import DataExplorer

class Container(containers.DeclarativeContainer):
    utils = providers.Singleton(Utils)
    model_training = providers.Factory(ModelTraining, utils=utils)
    data_explorer = providers.Factory(DataExplorer, utils=utils)

