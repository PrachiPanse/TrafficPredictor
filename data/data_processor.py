import numpy as np
import torch
import os

#################################################################################################
################### Function to calculate the mean and standard deviation #######################
#################################################################################################
def estimate_mean_std_dev(dataset_split, split_type):
    if split_type == 'train':
        exploded_dataframe = dataset_split.dataframe.explode(column='series_value')
        mean_value = exploded_dataframe['series_value'].mean()
        std_dev_value = exploded_dataframe['series_value'].std()
        print(f"mean = {mean_value}, std_dev = {std_dev_value}")
    return mean_value, std_dev_value

#################################################################################################
######################### Generate input and output pairs for the model #########################
#################################################################################################
def convert_to_input_output_list(input_series_value, lag_amount, forecast_horizon):
    len_total = len(input_series_value) 
    list_input_output_pairs = []
    for i in range(len_total - lag_amount - forecast_horizon + 1): #Generate samples for learning
        input_list = []
        output_list = []
        input_list = input_series_value[i : i + lag_amount]
        output_list = input_series_value[i + lag_amount : i + lag_amount + forecast_horizon]
        list_input_output_pairs.append([input_list, output_list])
    return list_input_output_pairs

#################################################################################################
########################### Function to preprocess the datasets #################################
#################################################################################################
def preprocess_datasets(dataset_split_dict, lag_amount, forecast_horizon, dataset_type_chosen):
    for i in ['train', 'val', 'test']:
        if i == 'train':
            mean_value, std_dev_value = estimate_mean_std_dev(dataset_split_dict[i], i)   
        #Normalization
        dataset_split_dict[i].dataframe['series_value_normalized']= dataset_split_dict[i].dataframe['series_value'].apply(lambda x: (np.asarray(x)-mean_value)/std_dev_value)
        #Generation of input output pairs
        dataset_split_dict[i].dataframe['input_output_lists'] = dataset_split_dict[i].dataframe['series_value_normalized'].apply(lambda x: convert_to_input_output_list(x, lag_amount, forecast_horizon))
        #Separate out the input and output pairs
        dataset_split_dict[i].dataframe = dataset_split_dict[i].dataframe.explode(column='input_output_lists')
        dataset_split_dict[i].dataframe['model_input_lists'] = dataset_split_dict[i].dataframe['input_output_lists'].apply(lambda x: x[0])
        dataset_split_dict[i].dataframe['model_output_lists'] = dataset_split_dict[i].dataframe['input_output_lists'].apply(lambda x: x[1])

    if 0:  
        dir_name = './logs/' + str(dataset_type_chosen)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dataset_split_dict['train'].dataframe.to_csv(os.path.join(dir_name, 'train_input_output_pairs.csv'),  columns = ['input_output_lists'])
        dataset_split_dict['val'].dataframe.to_csv(os.path.join(dir_name, 'val_input_output_pairs.csv'),  columns = ['input_output_lists'])
        dataset_split_dict['test'].dataframe.to_csv(os.path.join(dir_name, 'test_input_output_pairs.csv'),  columns = ['input_output_lists'])
 
    return dataset_split_dict, mean_value, std_dev_value

#################################################################################################
################### Function to setup dataloaders required by Pytorch models ####################
#################################################################################################
def setup_data_loaders(preprocessed_datasets_dict, batch_size):
    data_loaders = {}
    for i in ['train', 'val', 'test']:
        inputs_nparray = np.array(preprocessed_datasets_dict[i].dataframe['model_input_lists'].to_list())
        outputs_nparray = np.array(preprocessed_datasets_dict[i].dataframe['model_output_lists'].to_list())
        inputs_tensor = torch.FloatTensor(inputs_nparray)
        outputs_tensor = torch.FloatTensor(outputs_nparray)
        dataset = torch.utils.data.TensorDataset(inputs_tensor,outputs_tensor) # create your dataset
        data_loaders[i] = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True if i == 'train' else False) # create your dataloader
    return data_loaders

