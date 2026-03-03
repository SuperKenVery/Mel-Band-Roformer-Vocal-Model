import torch
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else ""

try:
    state_dict = torch.load("model.pt", map_location="cpu")
    print(f"Loaded model.pt with {len(state_dict)} keys.")
    
    count = 0
    for key in state_dict.keys():
        if pattern in key:
            print(key)
            count += 1
            if count > 20: # Limit output
                print("... (more)")
                break
            
except Exception as e:
    print(f"Error: {e}")
