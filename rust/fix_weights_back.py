import torch

try:
    state_dict = torch.load("model.pt", map_location="cpu")
    print(f"Loaded model.pt with {len(state_dict)} keys.")
    
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        
        # Rename norm.weight back to norm.gamma (because Burn RmsNorm uses gamma)
        if "norm.weight" in new_key:
            new_key = new_key.replace("norm.weight", "norm.gamma")
            
        # Ensure to_out is correct (it should be to_out.weight, which I did in previous step)
        # to_out.0.weight -> to_out.weight.
        # So I don't need to change to_out again if it's already correct.
        
        new_state_dict[new_key] = value
        
    print(f"Renamed keys. Saving to model.pt...")
    torch.save(new_state_dict, "model.pt")
    print("Done.")

except Exception as e:
    print(f"Error: {e}")
