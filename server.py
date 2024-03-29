import argparse
from data.data_reader import read_data
from common.types import DatasetType
from model.model import model_mapping_dict, Lightning_module
from user_interface.server_user_interface import create_user_interface

def get_parsed_arguments():
    #Creating an ArgumentParser object to store all information passed from command-line.
    parser = argparse.ArgumentParser(description='Server UI.') 
    #Dataset parameters
    parser.add_argument('--dataset_type', default = DatasetType.dataset_1_pedestrian_5_series, type=DatasetType, 
                        choices = list(DatasetType), help='Batch size', required = False)
    parser.add_argument('--lag_amount', default = 210, type=int, help='Amount of lag to be used for learning', required = False)
    parser.add_argument('--forecast_horizon_input', default = 24, type=int, help='Input forecasting horizon', required = False)
    #Model parameters
    parser.add_argument('--model_type', default = 'Linear_model', type=str, help='Model Type', 
                        choices = model_mapping_dict.keys(), required = False)
    parser.add_argument('--model_checkpoint', default = 'checkpoints/Linear_model/best_checkpoint.ckpt', 
                        type=str, help='Path for model checkpoint', required = False)
    args = parser.parse_args()
    return args

def main():
    parsed_args = get_parsed_arguments()
    print(f'\033[1m \nParsed data: \033[0m {parsed_args}')

    #Load the chosen dataset
    raw_dataset = read_data(dataset_type_chosen = parsed_args.dataset_type, verbose = True, print_enable = True)
    
    #Create dataset to be used for forecasting
    dataframe_forecast = raw_dataset.create_forecast_data(parsed_args.lag_amount)

    #Load model from saved checkpoint
    print(f"\033[1m \nLoading the trained model \033[0m")
    model = Lightning_module.load_from_checkpoint(
        checkpoint_path=parsed_args.model_checkpoint,
        model_type=parsed_args.model_type,
        lag_amount=parsed_args.lag_amount, 
        forecast_horizon=parsed_args.forecast_horizon_input,
        optimizer_type= 'SGD',
        learning_rate=0.001,
        loss_function='mse_loss'
    )

    model.eval() # Switch the model mode from training to inference (or evaluation) 
    model.freeze() #Freeze all params for inference
  
    #Make predictions
    print(f"\033[1m \nPredicting with the model \033[0m")   
    dataframe_forecast["prediction"] = dataframe_forecast["forecast_input"].apply(model.forecast)  #passing one 1-D numpy array at a time to the function
    print(f"dataframe_forecast['prediction'] =\n {dataframe_forecast['prediction']}")
    
    create_user_interface(dataframe_forecast)




if __name__ == "__main__":
    main()
