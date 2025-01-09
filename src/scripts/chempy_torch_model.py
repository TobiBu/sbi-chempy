import torch

# ----- Define the model ------------------------------------------------------------------------------------------------------------------------------------------

class Model_Torch(torch.nn.Module):
    def __init__(self, x_shape, y_shape):
        super(Model_Torch, self).__init__()
        self.l1 = torch.nn.Linear(x_shape, 100)
        self.l2 = torch.nn.Linear(100, 40)
        self.l3 = torch.nn.Linear(40, y_shape)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x