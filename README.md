# TrafficPredictor
 Repository to train and evaluate models for pedestrian traffic forecasting along with code for a user interface to access the same.

## Training models
To train a Linear model, run the following command.
```bash
python .\train_eval.py --model_type Linear_model --dataset_type dataset_1_pedestrian_small --num_epoch 5 --optimizer_type Adam --learning_rate 1e-3 --batch_size 16 --loss_function mse_loss    --forecast_horizon_input 24
```
For other models, update the ```model_type``` to the required value.


## Evaluating models
Run the following command for evaluation of models.
```bash
python .\train_eval.py --model_type Linear_model --dataset_type dataset_1_pedestrian_small --batch_size 16 --loss_function mse_loss    --forecast_horizon_input 24 --evaluation_mode --model_checkpoint checkpoints/Linear_model/best_checkpoint.ckpt
```
Replace ```model_checkpoint``` argument with the model to be evaluated.

## Running the streamlit server for UI
Run the following command to launch streamlit app.
```bash
streamlit run .\server.py
```
