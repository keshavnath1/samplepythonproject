{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "initial_id",
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "82cc5196187c0086",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2023-11-10T01:58:11.985236200Z",
     "start_time": "2023-11-10T01:58:09.731581900Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor([[-0.1885],\n",
      "        [-0.1793],\n",
      "        [-0.1829],\n",
      "        [-0.1787],\n",
      "        [-0.1785],\n",
      "        [-0.1834],\n",
      "        [-0.1791],\n",
      "        [-0.1843],\n",
      "        [-0.1762],\n",
      "        [-0.1864],\n",
      "        [-0.1814],\n",
      "        [-0.1858],\n",
      "        [-0.1789],\n",
      "        [-0.1698],\n",
      "        [-0.1667],\n",
      "        [-0.1912],\n",
      "        [-0.1683],\n",
      "        [-0.1766],\n",
      "        [-0.1665],\n",
      "        [-0.1832],\n",
      "        [-0.1766],\n",
      "        [-0.1826],\n",
      "        [-0.1848],\n",
      "        [-0.1892],\n",
      "        [-0.1802],\n",
      "        [-0.1824],\n",
      "        [-0.1773],\n",
      "        [-0.1891],\n",
      "        [-0.1756],\n",
      "        [-0.1823],\n",
      "        [-0.1777],\n",
      "        [-0.1706]], grad_fn=<AddmmBackward0>)\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "class TemporalModel(nn.Module):\n",
    "    def __init__(self, input_size, hidden_size, num_layers, output_size):\n",
    "        super(TemporalModel, self).__init__()\n",
    "        self.hidden_size = hidden_size\n",
    "        self.num_layers = num_layers\n",
    "        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)\n",
    "        self.fc = nn.Linear(hidden_size, output_size)\n",
    "\n",
    "    def forward(self, x):\n",
    "        # Initialize hidden state with zeros\n",
    "        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)\n",
    "        # Initialize cell state\n",
    "        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)\n",
    "        # We need to detach as we are doing truncated backpropagation through time (BPTT)\n",
    "        # If we don't, we'll backprop all the way to the start even after going through another batch\n",
    "        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))\n",
    "        # Decode the hidden state of the last time step\n",
    "        out = self.fc(out[:, -1, :])\n",
    "        return out\n",
    "\n",
    "# Define the model\n",
    "input_size = 10  # Number of features\n",
    "hidden_size = 50  # Number of features in hidden state\n",
    "num_layers = 2  # Number of stacked LSTM layers\n",
    "output_size = 1  # Number of output classes\n",
    "\n",
    "model = TemporalModel(input_size, hidden_size, num_layers, output_size)\n",
    "\n",
    "# Example of input data (batch_size, sequence_length, input_size)\n",
    "x = torch.randn(32, 5, input_size)\n",
    "\n",
    "# Forward pass\n",
    "output = model(x)\n",
    "print(output)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c3ed6f1a7e81df63",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m============================= test session starts =============================\u001b[0m\n",
      "platform win32 -- Python 3.8.18, pytest-7.4.0, pluggy-1.0.0\n",
      "rootdir: C:\\Users\\kesha\\workspace\\pythonProject\n",
      "plugins: anyio-3.5.0\n",
      "tensor([[-0.0330],\n",
      "        [-0.0579],\n",
      "        [-0.0499],\n",
      "        [-0.0469],\n",
      "        [-0.0670],\n",
      "        [-0.0418],\n",
      "        [-0.0511],\n",
      "        [-0.0337],\n",
      "        [-0.0547],\n",
      "        [-0.0659],\n",
      "        [-0.0505],\n",
      "        [-0.0654],\n",
      "        [-0.0472],\n",
      "        [-0.0695],\n",
      "        [-0.0350],\n",
      "        [-0.0400],\n",
      "        [-0.0438],\n",
      "        [-0.0431],\n",
      "        [-0.0624],\n",
      "        [-0.0533],\n",
      "        [-0.0381],\n",
      "        [-0.0473],\n",
      "        [-0.0569],\n",
      "        [-0.0597],\n",
      "        [-0.0624],\n",
      "        [-0.0555],\n",
      "        [-0.0481],\n",
      "        [-0.0578],\n",
      "        [-0.0512],\n",
      "        [-0.0315],\n",
      "        [-0.0453],\n",
      "        [-0.0527]], grad_fn=<AddmmBackward0>)\n",
      "collected 1 item\n",
      "\n",
      "test_temporal_model.py \u001b[32m.\u001b[0m\n",
      "\n",
      "\u001b[32m============================== \u001b[32m\u001b[1m1 passed\u001b[0m\u001b[32m in 1.67s\u001b[0m\u001b[32m ==============================\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "!pytest test_temporal_model.py -s\n",
    "\n"
   ]
  },
  {
   "cell_type": "raw",
   "id": "7ca33151",
   "metadata": {},
   "source": [
    "pytest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b17b84e9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
