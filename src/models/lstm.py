from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size,output_size, projection_size=32, hidden_size=128, num_layers=2):
        super().__init__()
        
        # input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, projection_size),
            nn.BatchNorm1d(projection_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # input projection
        x = x.view(batch_size * seq_len, -1)
        x = self.input_proj(x)
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM
        out, _ = self.lstm(x)
        
        # get the last time step output
        out = out[:, -1, :]
        
        # output layer
        out = self.output_layer(out)
        
        return out