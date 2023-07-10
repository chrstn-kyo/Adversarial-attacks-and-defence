import torch

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    
def load_model(Model_class, kwargs, file_path):
    model = Model_class(**kwargs)
    model.load_state_dict(torch.load(file_path))
    return model