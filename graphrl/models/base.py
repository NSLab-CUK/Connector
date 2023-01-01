import torch
import torch.nn as nn
import os
import json

class BaseModel(nn.Module):
    def __init__(self, device="cpu"):
        super(BaseModel, self).__init__()
        self.device = device
        
    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(os.path.join(file_path)))
        self.eval()

    def load_parameters(self, file_path):
        try:
            file = open(file_path, "r")
            parameters = json.loads(file.read())
            file.close()
            
            for param in parameters:
                parameters[param] = torch.Tensor(parameters[param], device=self.device)
            
            torch.load_state_dict(parameters, strict = False)
            self.eval()
        except:
            print('Error: cannot load parameters file.')
            exit()

    def save_parameters(self, file_path):
        try:
            file = open(file_path, 'w')
            file.write(json.dumps(self.convert_parameters_to_json()))
            file.close()
        except:
            print('Error: cannot save parameters file.')
            exit()

    def convert_parameters_to_json(self):
        parameters = self.state_dict()
        for param in parameters.keys():
            parameters[param] = parameters[param].cpu().numpy()

        return parameters


    def set_parameters(self, parameters):
        for param in parameters:
            parameters[param] = torch.Tensor(parameters[param], device=self.device)
        self.load_state_dict(parameters, strict = False)
        self.eval()
    

    

    


    


