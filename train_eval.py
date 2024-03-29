import argparse
from data.data_reader import read_data
from data.data_processor import preprocess_datasets, setup_data_loaders
from common.types import DatasetType
from model.model import model_mapping_dict, Lightning_module
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def get_parsed_arguments():
    #Creating an ArgumentParser object to store all information passed from command-line.
    parser = argparse.ArgumentParser(description='Training the model.') 
    #Dataset parameters
    parser.add_argument('--dataset_type', default = DatasetType.dataset_1_pedestrian_small, type=DatasetType, 
                        choices = list(DatasetType), help='Type of dataset used', required = False)
    parser.add_argument('--train_ratio', default = 60, type=int, help='Percentage data to be used for training', required = False)
    parser.add_argument('--val_ratio', default = 20, type=int, help='Percentage data to be used for validation', required = False)
    parser.add_argument('--test_ratio', default = 20, type=int, help='Percentage data to be used for testing', required = False)
    parser.add_argument('--lag_amount', default = 210, type=int, help='Amount of lag to be used for learning', required = False)
    parser.add_argument('--forecast_horizon_input', default = 24, type=int, help='Input forecasting horizon', required = False)

    #Model parameters
    parser.add_argument('--model_type', default = 'Linear_model', type=str, help='Model Type', 
                        choices = model_mapping_dict.keys(), required = False)
    parser.add_argument('--num_epoch', default = 5, type=int, help='Number of Epochs', required = False)
    parser.add_argument('--batch_size', default = 16, type=int, help='Batch size', required = False)
    parser.add_argument('--learning_rate', default = 0.001, type=float, help='Learning rate', required = False)
    parser.add_argument('--optimizer_type', default = 'SGD', type=str, choices = ['SGD', 'Adam'], 
                        help='Type of optimizer', required = False)
    parser.add_argument('--loss_function', default = 'mae_loss', type=str, choices = ['mae_loss', 'mse_loss'], 
                        help='Type of Loss function', required = False)
    parser.add_argument('--evaluation_mode', help='Specify argument to enable evaluation mode; Default is training mode', 
                        action='store_true')
    parser.add_argument('--model_checkpoint', default = None, type=str, help='Path for model checkpoint', required = False)
    args = parser.parse_args()
    return args


def main():
    parsed_args = get_parsed_arguments()
    print(f'\033[1m \n-----Parsed data: \033[0m {parsed_args}')

    #Load the chosen dataset
    raw_dataset = read_data(
        dataset_type_chosen = parsed_args.dataset_type, 
        verbose = False, 
        print_enable = False
    )
   
    #Split dataset into train, val, and test sets 
    dataset_split_dict = raw_dataset.get_train_test_split(
        train_ratio = parsed_args.train_ratio, 
        val_ratio = parsed_args.val_ratio, 
        test_ratio = parsed_args.test_ratio 
    )

    #Dataset preprocessing
    preprocessed_datasets_dict, mean_value, std_dev_value = preprocess_datasets(
        dataset_split_dict, 
        parsed_args.lag_amount, 
        parsed_args.forecast_horizon_input, 
        dataset_type_chosen = parsed_args.dataset_type
    )
    
    data_loaders = setup_data_loaders(preprocessed_datasets_dict, batch_size= parsed_args.batch_size)

    #Define the model
    print(f"\033[1m \n-----Define the model: \033[0m")
    seed_everything(seed=96, workers=True)
    model = Lightning_module(
        parsed_args.model_type, 
        parsed_args.lag_amount, 
        parsed_args.forecast_horizon_input,
        parsed_args.optimizer_type,
        learning_rate=parsed_args.learning_rate,
        loss_function = parsed_args.loss_function
    )
    
    logger = CSVLogger(
        save_dir= "logs/"+str(parsed_args.dataset_type)+"/lightning_logs/",
        name=parsed_args.model_type
    )

    #Checkpointing following https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=None,
        filename="best_checkpoint",
    )

    trainer = L.Trainer(
        max_epochs=parsed_args.num_epoch, 
        check_val_every_n_epoch=1, 
        log_every_n_steps=1000, 
        deterministic=True, 
        enable_checkpointing=True, 
        logger=logger, 
        callbacks=[checkpoint_callback],
        reload_dataloaders_every_n_epochs = 1
    )

    #Training
    if parsed_args.evaluation_mode == False:
        model.set_preprocessing_parameters(mean_value, std_dev_value)
        if parsed_args.model_type not in ['Predict_last_value', 'Predict_mean_value'] :
            print(f"\033[1m \n-----Training the model: \033[0m")
            trainer.fit(model, data_loaders['train'], data_loaders['val'])

    else: #Load model from saved checkpoint
        print(f"load the trained model:")
        model = Lightning_module.load_from_checkpoint(
            checkpoint_path=parsed_args.model_checkpoint,
            model_type=parsed_args.model_type,
            lag_amount=parsed_args.lag_amount, 
            forecast_horizon=parsed_args.forecast_horizon_input,
            optimizer_type= parsed_args.optimizer_type,
            learning_rate=parsed_args.learning_rate,
            loss_function=parsed_args.loss_function
            )
        model.eval() # Switch the model mode from training to inference (or evaluation) 
        model.freeze() #Freeze all params for inference

    #Validation
    print(f"\033[1m \n-----Validating the model at the end of training: \033[0m")
    trainer.validate(model, dataloaders=data_loaders['val'], ckpt_path=None if parsed_args.model_type in ['Predict_last_value', 'Predict_mean_value'] or parsed_args.evaluation_mode else 'best')

    #Testing
    print(f"\033[1m \n-----Testing the trained model: \033[0m")
    trainer.test(model, dataloaders=data_loaders['test'], ckpt_path=None if parsed_args.model_type in ['Predict_last_value', 'Predict_mean_value'] or parsed_args.evaluation_mode else 'best')

    print(f"------------------")

if __name__ == "__main__":
    main()