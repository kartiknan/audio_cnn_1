
import torch
import torch.nn.functional as F

def flatten_col(output):  return output.view((output.shape[0], -1))

def avg_pool_col(output): return flatten_col(F.adaptive_avg_pool2d(output, 1))

class ActGetter():
    
    '''
    Returns activations for chosen layers of a network.
    '''
        
    def __init__(self, model, target_layers, collate_fn=avg_pool_col):
        
        self.model, self.target_layers, self.collate_fn = model, target_layers, collate_fn
        self.activation_dict = {}
        self.register_hooks()
        
    def __call__(self, input):        
                
        self.empty_hooks()  # Empties stored hook list.        
        self.model.eval()
        
        with torch.no_grad(): self.model(input)

        # return flatttened list 
        return torch.cat(self.activations, dim=-1)  
    
    def empty_hooks(self):  self.activations = []
    def clear_all_activations(self): self.activation_dict = {}

    def get_layer_activations(self):
        # concat + flatten + numpy all batch activations individually
        act_dict = self.activation_dict.copy()
        for a in act_dict: 
            activations = torch.cat(act_dict[a])
            act_dict[a] = activations.view(activations.size(0), -1)

        # return in list form 
        return list(act_dict.keys()), act_dict
        
    def register_hooks(self):
        
        self.empty_hooks()
        
        def forward_hook(module, input, output):
            # append to list 
            self.activations.append(self.collate_fn(output).cpu())

            # append to consistent dictionary 
            if str(output.shape) in self.activation_dict:
                self.activation_dict[str(output.shape)].append(output)
            else:
                self.activation_dict[str(output.shape)] = [output]
            return None
        
        for layer in self.target_layers:
            layer.register_forward_hook(forward_hook)


