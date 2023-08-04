
from data_processing import DataProcessingFacade
from enums.process_type import ProcessType

if __name__ == "__main__":
    facade = DataProcessingFacade()
    type_process = 'TRAIN_EVALUATE'
    facade.run(type_process)
