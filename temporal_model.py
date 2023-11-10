#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TemporalModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Define the model
input_size = 10  # Number of features
hidden_size = 50  # Number of features in hidden state
num_layers = 2  # Number of stacked LSTM layers
output_size = 1  # Number of output classes

model = TemporalModel(input_size, hidden_size, num_layers, output_size)

# Example of input data (batch_size, sequence_length, input_size)
x = torch.randn(32, 5, input_size)

# Forward pass
output = model(x)
print(output)


# In[4]:


get_ipython().system('pytest test_temporal_model.py -s')


pytest
# In[ ]:




