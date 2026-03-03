import torch

try:
    state_dict = torch.load("model.pt", map_location="cpu")
    print(f"Loaded model.pt with {len(state_dict)} keys.")
    
    for key in state_dict.keys():
        if "mask_estimators" in key:
            print(key)
            
except Exception as e:
    print(f"Error: {e}")
