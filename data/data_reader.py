import pandas as pd
from common.types import dataset_mapping_dict
from data.data_load_Monash import convert_tsf_to_dataframe
import copy 
import os

#################################################################################################
########################### RawDataset Class and its functions ##################################
#################################################################################################
class RawDataset:
    #Class Attributes     

    #Initializer
    def __init__(self, dataset_type_chosen, dataframe, frequency, forecast_horizon, contain_missing_values, contain_equal_length):
        self.dataframe = dataframe
        self.dataset_type_chosen = dataset_type_chosen
        self.frequency = frequency
        self.forecast_horizon = forecast_horizon
        self.contain_missing_values = contain_missing_values
        self.contain_equal_length = contain_equal_length

    # Function to display a brief summary of the Dataset chosen
    def display_summary(self):
        print(f'\033[1m \n--------Dataset summary: \033[0m')
        print(f'dataset_type_chosen: {self.dataset_type_chosen} ')
        print(f'frequency: {self.frequency} ')
        print(f'forecast_horizon: {self.forecast_horizon} ')
        print(f'contain_missing_values: {self.contain_missing_values} ')
        print(f'contain_equal_length: {self.contain_equal_length}  ')
        print(f'loaded_data.axes= {self.dataframe.axes}')
        print(f'loaded_data.shape= {self.dataframe.shape}')

    def save(self, data_identifier):
        dir_name = './logs/' + str(self.dataset_type_chosen)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        self.dataframe.to_csv(os.path.join(dir_name, f'{data_identifier}_dataset.csv'), index = False)
       
    def print_dataset(self, data_identifier):
        print(f"{data_identifier} dataset:\n {self.dataframe}")

    #Function to split the raw dataset into train, val and test datasets 
    def get_train_test_split(self, train_ratio, val_ratio, test_ratio):
        assert train_ratio + val_ratio + test_ratio == 100, "Invalid train-val-test ratio input."

        dataframe_train = pd.DataFrame(columns= self.dataframe.columns, index = self.dataframe.index)
        dataframe_val = pd.DataFrame(columns= self.dataframe.columns, index = self.dataframe.index)
        dataframe_test = pd.DataFrame(columns= self.dataframe.columns, index = self.dataframe.index)
        
        for i in range(self.dataframe.shape[0]):
            num_total = len(self.dataframe.series_value[i])
            num_train = int(num_total*train_ratio/100)
            num_val = int(num_total*val_ratio/100)

            dataframe_train.iloc[i, :].at['series_value'] = self.dataframe.iloc[i, :].at['series_value'][0 : num_train]
            dataframe_val.iloc[i, :].at['series_value'] = self.dataframe.iloc[i, :].at['series_value'][num_train : (num_train + num_val)]
            dataframe_test.iloc[i, :].at['series_value'] = self.dataframe.iloc[i, :].at['series_value'][(num_train + num_val) : num_total]

        dataframe_train.series_name = self.dataframe.series_name
        dataframe_train.start_timestamp = self.dataframe.start_timestamp   #TODO timestamp can be calculated and assigned per split for each series
        dataframe_val.series_name = self.dataframe.series_name
        dataframe_val.start_timestamp = self.dataframe.start_timestamp
        dataframe_test.series_name = self.dataframe.series_name
        dataframe_test.start_timestamp = self.dataframe.start_timestamp

        raw_dataset_train = copy.deepcopy(self)
        raw_dataset_train.dataframe = dataframe_train
        raw_dataset_val = copy.deepcopy(self)
        raw_dataset_val.dataframe = dataframe_val
        raw_dataset_test = copy.deepcopy(self)
        raw_dataset_test.dataframe = dataframe_test

        dataset_split_dict = {'train': raw_dataset_train, 
                              'val': raw_dataset_val, 
                              'test': raw_dataset_test }

        if 0:
            for split_name, split_data in dataset_split_dict.items():
                split_data.save(split_name)

        if 0:
            for split_name, split_data in dataset_split_dict.items():
                split_data.print_dataset(split_name)

        return dataset_split_dict


    def create_forecast_data(self, lag_amount):
        dataframe_forecast = pd.DataFrame(columns= self.dataframe.columns, index = self.dataframe.index)
        dataframe_forecast['forecast_input'] = self.dataframe['series_value'].apply(lambda x: x[len(x) - lag_amount :])
        dataframe_forecast.series_name = self.dataframe.series_name
        dataframe_forecast.start_timestamp = self.dataframe.start_timestamp           
        return dataframe_forecast
    

#################################################################################################
########################### Function to load the chosen dataset #################################
#################################################################################################
def read_data(dataset_type_chosen, verbose = False, print_enable = False):
    dataset_path = dataset_mapping_dict[dataset_type_chosen]
    print(f'\nChosen dataset path: {dataset_path}')

    # Load the dataset
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(dataset_path, limit=100)
    
    raw_dataset = RawDataset(
        dataset_type_chosen = dataset_type_chosen,
        dataframe = loaded_data,
        frequency = frequency,
        forecast_horizon = forecast_horizon,
        contain_missing_values = contain_missing_values,
        contain_equal_length = contain_equal_length
    )

    if verbose:
        raw_dataset.display_summary()       

    if print_enable:
        print(f'\nraw_dataset:\n {raw_dataset.dataframe}')

    return raw_dataset















def main():
    dataset_dict = read_data('../datasets/2_pedestrian_counts/pedestrian_counts_dataset.tsf')  
    return None

if __name__ == "__main__":
    main()