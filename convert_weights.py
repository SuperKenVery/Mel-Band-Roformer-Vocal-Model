import torch
import re

def convert_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        
        # RoformerLayer: tuple (0, 1) -> named (time_transformer, freq_transformer)
        new_key = re.sub(r'layers\.(\d+)\.0\.', r'layers.\1.time_transformer.', new_key)
        new_key = re.sub(r'layers\.(\d+)\.1\.', r'layers.\1.freq_transformer.', new_key)
        
        # BandFeature: tuple (0, 1) -> named (norm, linear)
        new_key = re.sub(r'band_split\.to_features\.(\d+)\.0\.', r'band_split.to_features.\1.norm.', new_key)
        new_key = re.sub(r'band_split\.to_features\.(\d+)\.1\.', r'band_split.to_features.\1.linear.', new_key)
        
        # MLP: tuple (0) -> named (layers)
        # mask_estimators.X.to_freqs.Y.0... -> mask_estimators.X.to_freqs.Y.layers...
        new_key = re.sub(r'mask_estimators\.(\d+)\.to_freqs\.(\d+)\.0\.', r'mask_estimators.\1.to_freqs.\2.layers.', new_key)
        
        # OutputBlock: tuple (0) -> named (linear)
        new_key = re.sub(r'\.to_out\.0\.', r'.to_out.linear.', new_key)
        
        # FFSequential: tuple (0, 1, 4) -> named (norm, linear1, linear2)
        # Matches net.0, net.1, net.4
        new_key = re.sub(r'\.net\.0\.', r'.net.norm.', new_key)
        new_key = re.sub(r'\.net\.1\.', r'.net.linear1.', new_key)
        new_key = re.sub(r'\.net\.4\.', r'.net.linear2.', new_key)
        
        print(f"{key} -> {new_key}")
        new_state_dict[new_key] = value
        
    return new_state_dict

if __name__ == "__main__":
    try:
        sd = torch.load("rust/model.pt", map_location="cpu")
        new_sd = convert_keys(sd)
        torch.save(new_sd, "rust/model_fixed.pt")
        print("Converted model saved to rust/model_fixed.pt")
    except Exception as e:
        print(f"Error: {e}")
