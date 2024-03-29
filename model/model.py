import torch
import lightning as L
import torch.nn as nn
from torch import optim
import math

#################################################################################################
##################### Baseline - Trivial model predicts last value ##############################
#################################################################################################
class Predict_last_value(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__() 

    def forward(self, input):
        pred = input[:, -1]
        return pred
    
#################################################################################################
##################### Baseline - Trivial model predicts mean value ##############################
#################################################################################################
class Predict_mean_value(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()

    def forward(self, input):
        pred = torch.mean(input, axis = 1)
        return pred
    
#################################################################################################
############################################ Linear model #######################################
#################################################################################################
class Linear_model(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features = lag_amount, out_features = forecast_horizon)

    def forward(self, input):
        pred = self.linear_layer(input)
        return pred  


#################################################################################################
###################### Feedforward Neural Network (FNN) model ###################################
#################################################################################################
def get_fnn_model(lag_amount, forecast_horizon):
    return  nn.Sequential(nn.Linear(in_features = lag_amount, out_features = 100),
                          nn.ReLU(),
                          nn.Linear(in_features = 100, out_features = forecast_horizon)
                          )

class FNN_model_2(torch.nn.Module): #FNN with skip connections
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        hidden_size = 100
        self.linear_layer_1 = nn.Linear(in_features = lag_amount, out_features = hidden_size)
        self.relu_layer = nn.ReLU()
        self.linear_layer_2 = nn.Linear(in_features = hidden_size + lag_amount, out_features = forecast_horizon)

    def forward(self, input): #input size: ([batch_size, lag])
        hidden = self.linear_layer_1(input) # hidden size: ([batch_size, hidden_size])
        hidden = self.relu_layer(hidden) # hidden size: ([batch_size, hidden_size])
        hidden = torch.cat([hidden, input], dim = 1)  # hidden size: ([batch_size, 310])
        pred = self.linear_layer_2(hidden) #pred size: torch.Size([batch_size, forecast_horizon])
        return pred 
    
    

#################################################################################################
########################### Convolutional Neural Network (CNN) model ############################
#################################################################################################
class CNN_model(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        kernel_1_size = 12
        num_channels = 10
        conv_output_size = lag_amount-kernel_1_size+1
        pooling_output_size = math.floor((conv_output_size-kernel_1_size-1)/12 + 1)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = num_channels, stride = 1, kernel_size = kernel_1_size)
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_1_size, stride=kernel_1_size)
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_1 = nn.Linear(
            in_features = pooling_output_size*num_channels,
            out_features = 32, bias=True
            )
        self.relu_layer = nn.ReLU()
        self.linear_2= nn.Linear(in_features = 32, out_features = forecast_horizon)

    def forward(self, input):
        hidden = self.conv(input.unsqueeze(1))
        hidden = self.max_pooling(hidden)
        hidden = self.flatten_layer(hidden)
        hidden = self.relu_layer(hidden)
        hidden = self.linear_1(hidden)
        hidden = self.relu_layer(hidden)
        pred = self.linear_2(hidden)
        return pred


class CNN_model_2(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        kernel_1_size = 24
        num_channels = 48 
        conv_stride = 24
        dilation = 1
        padding = 0
        conv_output_size = math.floor(((lag_amount +2 * padding - dilation *(kernel_1_size - 1) - 1)/conv_stride) + 1)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = num_channels, stride = conv_stride, kernel_size = kernel_1_size)
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_1 = nn.Linear(
            in_features = conv_output_size*num_channels,
            out_features = 32, bias=True
            )
        self.relu_layer = nn.ReLU()
        self.linear_2= nn.Linear(in_features = 32, out_features = forecast_horizon)

    def forward(self, input):
        hidden = self.conv(input.unsqueeze(1))
        hidden = self.flatten_layer(hidden)
        hidden = self.relu_layer(hidden)
        hidden = self.linear_1(hidden)
        hidden = self.relu_layer(hidden)
        pred = self.linear_2(hidden)
        return pred
    





class CNN_model_3(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        kernel_1_size = 24
        num_channels = 48 
        conv_stride = 24
        dilation = 1
        padding = 0
        conv_output_size = math.floor(((lag_amount +2 * padding - dilation *(kernel_1_size - 1) - 1)/conv_stride) + 1)
        self.conv = nn.Conv1d(in_channels = 1, out_channels = num_channels, stride = conv_stride, kernel_size = kernel_1_size)
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_1 = nn.Linear(
            in_features = conv_output_size*num_channels,
            out_features = 32, bias=True
            )
        self.relu_layer = nn.ReLU()
        self.linear_2= nn.Linear(in_features = 32 + lag_amount, out_features = forecast_horizon)

    def forward(self, input):
        hidden = self.conv(input.unsqueeze(1))
        hidden = self.flatten_layer(hidden)
        hidden = self.relu_layer(hidden)
        hidden = self.linear_1(hidden)
        hidden = self.relu_layer(hidden)
        hidden = torch.cat([hidden, input], dim = 1)
        pred = self.linear_2(hidden)
        return pred
    
#################################################################################################
####################################### Transformer model #######################################
#################################################################################################
class Transformer_model(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon): #input size is batch size * sequence length (210 here)
        super().__init__()
        self.num_heads = 1      #num of ways in which attention is done
        self.num_layers = 3
        self.dim_feedforward = 18
        self.position_embedding_dimension = 5
        self.num_features = 1+self.position_embedding_dimension #(size of each vector) which is equal to input + position features
        self.position_embedding_layer = nn.Embedding(num_embeddings = lag_amount, embedding_dim = self.position_embedding_dimension)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_features, 
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=self.dim_feedforward, 
                                                                    batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
        self.relu_layer = nn.ReLU()
        self.linear= nn.Linear(in_features = self.num_features * lag_amount, out_features = forecast_horizon) #input features to be selected using techniques like - 1. mean over seq len dimension 2. max pooling (usually better) 3. select any particular one (last might be better for time series)
         
    def forward(self, input): #input size: batch_size * seq_len  #model expects batch_size*seqlen*num_features
        position_values = torch.arange(input.shape[1], device=input.device) #size: seq_len
        position_values = position_values.repeat(input.shape[0], 1) #size: batch_size*seq_len
        position_values_embedding = self.position_embedding_layer(position_values) #size: batch_size*seq_len*position_embedding_dimension
        input_with_positions = torch.cat([input.unsqueeze(2), position_values_embedding], dim=2) #size: batch_size*seq_len*num_features          #these can be added or concatenated
        hidden = self.transformer_encoder(input_with_positions)  #size: batch_size*seq_len*num_features
        hidden = hidden + input_with_positions  #skip connection
        hidden = self.relu_layer(hidden) #size: batch_size*seq_len*num_features
        hidden = hidden.flatten(start_dim = 1)  #size: batch_size*(seq_len x num_features)
        pred = self.linear(hidden)#size: batch_size*FH
        return pred
    



class Transformer_model_2(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon): #input size is batch size * sequence length (210 here)
        super().__init__()
        self.num_heads = 1      #num of ways in which attention is done
        self.num_layers = 3
        self.dim_feedforward = 18
        self.position_embedding_dimension = 3
        self.num_features = 1+self.position_embedding_dimension #(size of each vector) which is equal to input + position features
        self.position_embedding_layer = nn.Embedding(num_embeddings = lag_amount, embedding_dim = self.position_embedding_dimension)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_features, 
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=self.dim_feedforward, 
                                                                    batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
        self.relu_layer = nn.ReLU()
        self.linear= nn.Linear(in_features = self.num_features * lag_amount, out_features = forecast_horizon) #input features to be selected using techniques like - 1. mean over seq len dimension 2. max pooling (usually better) 3. select any particular one (last might be better for time series)
         
    def forward(self, input): #input size: batch_size * seq_len  #model expects batch_size*seqlen*num_features
        position_values = torch.arange(input.shape[1], device=input.device) #size: seq_len
        position_values = position_values.repeat(input.shape[0], 1) #size: batch_size*seq_len
        position_values_embedding = self.position_embedding_layer(position_values) #size: batch_size*seq_len*position_embedding_dimension
        input_with_positions = torch.cat([input.unsqueeze(2), position_values_embedding], dim=2) #size: batch_size*seq_len*num_features          #these can be added or concatenated
        hidden = self.transformer_encoder(input_with_positions)  #size: batch_size*seq_len*num_features
        hidden = hidden + input_with_positions  #skip connection
        hidden = self.relu_layer(hidden) #size: batch_size*seq_len*num_features
        hidden = hidden.flatten(start_dim = 1)  #size: batch_size*(seq_len x num_features)
        pred = self.linear(hidden)#size: batch_size*FH
        return pred
    




class Transformer_model_3(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon): #input size is batch size * sequence length (210 here)
        super().__init__()
        self.num_heads = 1      #num of ways in which attention is done
        self.num_layers = 3
        self.dim_feedforward = 18
        self.dropout_value = 0
        self.position_embedding_dimension = 3
        self.num_features = 1+self.position_embedding_dimension #(size of each vector) which is equal to input + position features
        self.position_embedding_layer = nn.Embedding(num_embeddings = lag_amount, embedding_dim = self.position_embedding_dimension)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_features, 
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=self.dim_feedforward, 
                                                                    batch_first = True,
                                                                    dropout = self.dropout_value)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
        self.relu_layer = nn.ReLU()
        self.linear= nn.Linear(in_features = self.num_features * lag_amount, out_features = forecast_horizon) #input features to be selected using techniques like - 1. mean over seq len dimension 2. max pooling (usually better) 3. select any particular one (last might be better for time series)

    def forward(self, input): #input size: batch_size * seq_len  #model expects batch_size*seqlen*num_features
        position_values = torch.arange(input.shape[1], device=input.device) #size: seq_len
        position_values = position_values.repeat(input.shape[0], 1) #size: batch_size*seq_len
        position_values_embedding = self.position_embedding_layer(position_values) #size: batch_size*seq_len*position_embedding_dimension
        input_with_positions = torch.cat([input.unsqueeze(2), position_values_embedding], dim=2) #size: batch_size*seq_len*num_features          #these can be added or concatenated
        hidden = self.transformer_encoder(input_with_positions)  #size: batch_size*seq_len*num_features
        hidden = hidden + input_with_positions  #skip connection
        hidden = self.relu_layer(hidden) #size: batch_size*seq_len*num_features
        hidden = hidden.flatten(start_dim = 1)  #size: batch_size*(seq_len x num_features)
        pred = self.linear(hidden)#size: batch_size*FH
        return pred
    


class Transformer_model_4(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon): #input size is batch size * sequence length (210 here)
        super().__init__()
        self.num_heads = 1      #num of ways in which attention is done
        self.num_layers = 3
        self.dim_feedforward = 18
        self.dropout_value = 0
        self.position_embedding_dimension = 3
        self.num_features = 1+self.position_embedding_dimension #(size of each vector) which is equal to input + position features
        self.position_embedding_layer = nn.Embedding(num_embeddings = lag_amount, embedding_dim = self.position_embedding_dimension)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_features, 
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=self.dim_feedforward, 
                                                                    batch_first = True,
                                                                    dropout = self.dropout_value)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
        self.relu_layer = nn.ReLU()
        self.linear= nn.Linear(in_features = self.num_features * lag_amount, out_features = forecast_horizon) #input features to be selected using techniques like - 1. mean over seq len dimension 2. max pooling (usually better) 3. select any particular one (last might be better for time series)

    def forward(self, input): #input size: batch_size * seq_len  #model expects batch_size*seqlen*num_features
        position_values = torch.arange(input.shape[1], device=input.device) #size: seq_len
        position_values = position_values.repeat(input.shape[0], 1) #size: batch_size*seq_len
        position_values_embedding = self.position_embedding_layer(position_values) #size: batch_size*seq_len*position_embedding_dimension
        input_with_positions = torch.cat([input.unsqueeze(2), position_values_embedding], dim=2) #size: batch_size*seq_len*num_features          #these can be added or concatenated
        hidden = self.transformer_encoder(input_with_positions)  #size: batch_size*seq_len*num_features
        hidden = hidden + input_with_positions  #skip connection
        hidden = hidden.flatten(start_dim = 1)  #size: batch_size*(seq_len x num_features)
        pred = self.linear(hidden)#size: batch_size*FH
        return pred
    


class Transformer_model_5(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon): #input size is batch size * sequence length (210 here)
        super().__init__()
        self.num_heads = 1      #num of ways in which attention is done
        self.num_layers = 3
        self.dim_feedforward = 18
        self.dropout_value = 0
        self.position_embedding_dimension = 3
        self.num_features = 1+self.position_embedding_dimension #(size of each vector) which is equal to input + position features
        self.position_embedding_layer = nn.Embedding(num_embeddings = lag_amount, embedding_dim = self.position_embedding_dimension)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_features, 
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=self.dim_feedforward, 
                                                                    batch_first = True,
                                                                    dropout = self.dropout_value)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
        self.relu_layer = nn.ReLU()
        self.linear= nn.Linear(in_features = self.num_features * lag_amount, out_features = forecast_horizon) #input features to be selected using techniques like - 1. mean over seq len dimension 2. max pooling (usually better) 3. select any particular one (last might be better for time series)

    def forward(self, input): #input size: batch_size * seq_len  #model expects batch_size*seqlen*num_features
        position_values = torch.arange(input.shape[1], device=input.device) #size: seq_len
        position_values = position_values.repeat(input.shape[0], 1) #size: batch_size*seq_len
        position_values_embedding = self.position_embedding_layer(position_values) #size: batch_size*seq_len*position_embedding_dimension
        input_with_positions = torch.cat([input.unsqueeze(2), position_values_embedding], dim=2) #size: batch_size*seq_len*num_features          #these can be added or concatenated
        hidden = self.transformer_encoder(input_with_positions)  #size: batch_size*seq_len*num_features
        hidden = self.relu_layer(hidden) #size: batch_size*seq_len*num_features
        hidden = hidden.flatten(start_dim = 1)  #size: batch_size*(seq_len x num_features)
        pred = self.linear(hidden)#size: batch_size*FH
        return pred




class Transformer_model_6(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon): #input size is batch size * sequence length (210 here)
        super().__init__()
        self.num_heads = 1      #num of ways in which attention is done
        self.num_layers = 3
        self.dim_feedforward = 18
        self.dropout_value = 0
        self.position_embedding_dimension = 3
        self.num_features = 1+self.position_embedding_dimension #(size of each vector) which is equal to input + position features
        self.position_embedding_layer = nn.Embedding(num_embeddings = lag_amount, embedding_dim = self.position_embedding_dimension)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.num_features, 
                                                                    nhead=self.num_heads, 
                                                                    dim_feedforward=self.dim_feedforward, 
                                                                    batch_first = True,
                                                                    dropout = self.dropout_value)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.num_layers)
        self.relu_layer = nn.ReLU()
        self.linear= nn.Linear(in_features = (self.num_features+1) * lag_amount, out_features = forecast_horizon) #input features to be selected using techniques like - 1. mean over seq len dimension 2. max pooling (usually better) 3. select any particular one (last might be better for time series)

    def forward(self, input): #input size: batch_size * seq_len  #model expects batch_size*seqlen*num_features
        position_values = torch.arange(input.shape[1], device=input.device) #size: seq_len
        position_values = position_values.repeat(input.shape[0], 1) #size: batch_size*seq_len
        position_values_embedding = self.position_embedding_layer(position_values) #size: batch_size*seq_len*position_embedding_dimension
        input_with_positions = torch.cat([input.unsqueeze(2), position_values_embedding], dim=2) #size: batch_size*seq_len*num_features          #these can be added or concatenated
        hidden = self.transformer_encoder(input_with_positions)  #size: batch_size*seq_len*num_features
        hidden = self.relu_layer(hidden) #size: batch_size*seq_len*num_features
        hidden = torch.cat([hidden, input.unsqueeze(2)], dim=2) #size: batch_size*seq_len*(num_features+1)
        hidden = hidden.flatten(start_dim = 1)  #size: batch_size*(seq_len x (num_features+1))
        pred = self.linear(hidden)#size: batch_size*FH
        return pred
    



#################################################################################################
################################ Recurrent Neural Network #######################################
#################################################################################################
class RNN_model(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        self.LSTM_layer = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.relu_layer = nn.ReLU()
        self.linear_layer = torch.nn.Linear(in_features = 100, out_features = forecast_horizon)

    def forward(self, input):
        hidden, (h_n, c_n) = self.LSTM_layer(input.unsqueeze(2))
        hidden = self.relu_layer(hidden)
        pred = self.linear_layer(hidden[:, -1, :]) #retaining only the last of the sequence
        return pred 
    

class RNN_model_2(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        self.LSTM_layer = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.relu_layer = nn.ReLU()
        self.linear_layer = torch.nn.Linear(in_features = 100, out_features = forecast_horizon)

    def forward(self, input):
        hidden, (h_n, c_n) = self.LSTM_layer(input.unsqueeze(2))
        pred = self.linear_layer(hidden[:, -1, :]) #retaining only the last of the sequence
        return pred 
    

class RNN_model_3(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        self.LSTM_layer = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.relu_layer = nn.ReLU()
        self.linear_layer = torch.nn.Linear(in_features = 100 + 1, out_features = forecast_horizon)

    def forward(self, input): #input size: ([batch_size, lag]) 
        hidden, (h_n, c_n) = self.LSTM_layer(input.unsqueeze(2)) #hidden size: ([batch_size, lag, hidden_size])
        hidden = self.relu_layer(hidden) #hidden size: ([batch_size, lag, hidden_size])
        hidden = torch.cat([hidden, input.unsqueeze(2)], dim = 2) #hidden size: ([batch_size, lag, hidden_size + 1])
        pred = self.linear_layer(hidden[:, -1, :]) #retaining only the last of the sequence
        return pred 


class RNN_model_4(torch.nn.Module):
    def __init__(self, lag_amount, forecast_horizon):
        super().__init__()
        self.LSTM_layer = nn.LSTM(input_size=1, hidden_size=100, num_layers=1, batch_first=True)
        self.relu_layer = nn.ReLU()
        self.linear_layer = torch.nn.Linear(in_features = 100 + lag_amount, out_features = forecast_horizon)

    def forward(self, input): #input size: ([batch_size, lag]) 
        hidden, (h_n, c_n) = self.LSTM_layer(input.unsqueeze(2)) #hidden size: ([batch_size, lag, hidden_size])
        hidden = self.relu_layer(hidden) #hidden size: ([batch_size, lag, hidden_size])
        hidden = hidden[:, -1, :]  #retaining only the last of the sequence
        hidden = torch.cat([hidden, input], dim = 1)
        pred = self.linear_layer(hidden) 
        return pred
       
#################################################################################################
########################### Lightning module ####################################################
#################################################################################################
class Lightning_module(L.LightningModule):
    def __init__(self, model_type, lag_amount, forecast_horizon, optimizer_type, learning_rate, loss_function):
        super().__init__()
        self.model_type = model_type
        self.lag_amount = lag_amount
        self.forecast_horizon = forecast_horizon
        self.model = model_mapping_dict[self.model_type](lag_amount, forecast_horizon)
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.loss_function = nn.functional.l1_loss if loss_function == 'mae_loss' else nn.functional.mse_loss

    def configure_optimizers(self):
        if self.optimizer_type == 'SGD':
            optimizer = optim.SGD(params = self.parameters(), lr=self.learning_rate, momentum=0, dampening=0, weight_decay=0)
        elif self.optimizer_type == 'Adam':
            optimizer = optim.Adam(params = self.parameters(), lr=self.learning_rate)
        return optimizer
        
    def training_step (self, batch, batch_idx):
        x_in, y_in = batch
        y = self.model(x_in)
        loss = self.loss_function(input = y, target = y_in, reduction='mean')
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_in, y_in = batch
        y = self.model(x_in)
        loss = self.loss_function(input = y, target = y_in, reduction='mean')
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x_in, y_in = batch
        y = self.model(x_in)
        loss = self.loss_function(input = y, target = y_in, reduction='mean')
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint['training_mean'] = self.training_mean
        checkpoint['training_std_dev'] = self.training_std_dev

    def on_load_checkpoint(self, checkpoint):
        self.training_mean = checkpoint['training_mean']
        self.training_std_dev = checkpoint['training_std_dev']

    def set_preprocessing_parameters(self, mean_value, std_dev_value):
        self.training_mean = mean_value
        self.training_std_dev = std_dev_value

    def forecast(self, input_time_series): #input_time_series is a 1D numpy array of lag length #(lag_amount,0)
        #preprocess
        input_time_series_normalized = (input_time_series-self.training_mean)/self.training_std_dev      #(lag_amount,)
        #numpy to tensor
        input_time_series_tensor = torch.FloatTensor(input_time_series_normalized) #torch.Size([lag_amount])
        #unsqueeze
        input_time_series_unsqueezed = input_time_series_tensor.unsqueeze(0) #torch.Size([1, lag_amount])
        #model call
        prediction_tensor = self.model(input_time_series_unsqueezed) #torch.Size([1, forecast_horizon])
        #squeeze
        prediction_tensor_squeezed = prediction_tensor.squeeze(0) #torch.Size([forecast_horizon])
        #tensor to numpy
        prediction_nparray = prediction_tensor_squeezed.numpy() #(forecast_horizon,)
        #postprocess
        prediction_normalized_nparray = prediction_nparray * self.training_std_dev + self.training_mean #(forecast_horizon,)
        prediction = prediction_normalized_nparray.astype(int) #(forecast_horizon,)
        prediction[prediction < 0 ] = 0 #clip to 0 here to avoid any negative predictions
        return prediction     
   
#################################################################################################
#################################### Model mapping dictionary ###################################
#################################################################################################
model_mapping_dict = {
    'Predict_last_value' : Predict_last_value,
    'Predict_mean_value' : Predict_mean_value,
    'Linear_model'       : Linear_model,
    'CNN_model'          : CNN_model,
    'RNN_model'          : RNN_model,
    'Transformer_model'  : Transformer_model,
    'FNN_model'          : get_fnn_model,
    'CNN_model_2'        : CNN_model_2,
    'Transformer_model_2': Transformer_model_2,
    'Transformer_model_3': Transformer_model_3,
    'Transformer_model_4': Transformer_model_4,
    'RNN_model_2'        : RNN_model_2,
    'Transformer_model_5': Transformer_model_5,
    'Transformer_model_6': Transformer_model_6,
    'CNN_model_3'        : CNN_model_3,
    'RNN_model_3'        : RNN_model_3,
    'FNN_model_2'        : FNN_model_2,
    'RNN_model_4'        : RNN_model_4,
}

