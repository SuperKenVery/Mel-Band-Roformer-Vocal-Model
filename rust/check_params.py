import torch

try:
    state_dict = torch.load("model.pt", map_location="cpu")
    if "stft_n_fft" in state_dict:
        print(f"stft_n_fft: {state_dict['stft_n_fft']}")
    else:
        print("stft_n_fft not found in state_dict")
        
    if "stft_hop_length" in state_dict:
         print(f"stft_hop_length: {state_dict['stft_hop_length']}")

except Exception as e:
    print(f"Error: {e}")
