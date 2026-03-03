import torch

try:
    state_dict = torch.load("model.pt", map_location="cpu")
    print(f"Loaded model.pt with {len(state_dict)} keys.")
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Rename norm.gamma to norm.weight
        if "norm.gamma" in new_key:
            new_key = new_key.replace("norm.gamma", "norm.weight")
            
        # Rename to_out.0 to to_out
        if "to_out.0." in new_key:
            new_key = new_key.replace("to_out.0.", "to_out.")
            
        new_state_dict[new_key] = value
        
    print(f"Renamed keys. Saving to model.pt...")
    torch.save(new_state_dict, "model.pt")
    print("Done.")

except Exception as e:
    print(f"Error: {e}")
