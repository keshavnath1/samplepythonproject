import torch
import pytest
from temporal_model import TemporalModel  # Assuming your model is in a file named temporal_model.py


def test_output_shape():
    input_size = 10  # Number of features
    hidden_size = 50  # Number of features in hidden state
    num_layers = 2  # Number of stacked LSTM layers
    output_size = 1  # Number of output classes
    batch_size = 32  # Batch size
    sequence_length = 5  # Length of the time series

    # Initialize the model
    model = TemporalModel(input_size, hidden_size, num_layers, output_size)

    # Create some example input data with the shape (batch_size, sequence_length, input_size)
    x = torch.randn(batch_size, sequence_length, input_size)

    # Get the model output
    output = model(x)

    # Check the output shape
    assert output.shape == (
    batch_size, output_size), f"Expected output shape of (batch_size, output_size), but got {output.shape}"


# You can add additional tests here, such as checking for the proper device assignment,
# the consistency of output on multiple forward passes with the same input, etc.

# This allows running the test from the command line
if __name__ == "__main__":
    pytest.main([__file__])
