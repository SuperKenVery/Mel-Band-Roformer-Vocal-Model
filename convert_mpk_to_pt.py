
import gzip
import msgpack
import torch
import numpy as np
import os

def load_mpk(path):
    print(f"Loading {path}...")
    try:
        # Try gzip first
        with gzip.open(path, "rb") as f:
            data = msgpack.unpack(f)
    except:
        # Try raw
        with open(path, "rb") as f:
            data = msgpack.unpack(f)
    return data

def extract_tensors_recursive(data, prefix, tensors):
    if isinstance(data, dict):
        # Check if it is a parameter tensor
        if "param" in data and isinstance(data["param"], dict):
            param = data["param"]
            # It seems Burn NamedMpk uses "value" or "bytes" for data?
            # My inspection showed "bytes".
            if "bytes" in param:
                b = param["bytes"]
                shape = param.get("shape", [])
                dtype = param.get("dtype", "F32")
                
                if dtype == "F32":
                    np_dtype = np.float32
                elif dtype == "F16":
                    np_dtype = np.float16
                elif dtype == "I64":
                    np_dtype = np.int64
                else:
                    print(f"Warning: Unknown dtype {dtype} for {prefix}")
                    np_dtype = np.float32
                    
                # Burn saves as LE bytes usually.
                # np.frombuffer interprets as machine endian (usually LE).
                arr = np.frombuffer(b, dtype=np_dtype).copy()
                if shape:
                    arr = arr.reshape(shape)
                
                t = torch.from_numpy(arr)
                tensors[prefix] = t
                return

        # Recurse for children
        for k, v in data.items():
            if k == "metadata" or k == "_b": continue
            
            new_prefix = f"{prefix}.{k}" if prefix else k
            extract_tensors_recursive(v, new_prefix, tensors)
            
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_prefix = f"{prefix}.{i}" if prefix else str(i)
            extract_tensors_recursive(v, new_prefix, tensors)

def main():
    mpk_path = "rust/model.mpk"
    if not os.path.exists(mpk_path):
        print(f"File not found: {mpk_path}")
        return
        
    data = load_mpk(mpk_path)
    item = data.get("item")
    if not item:
        print("No item found in mpk")
        return
        
    print("Extracting tensors...")
    tensors = {}
    extract_tensors_recursive(item, "", tensors)
    print(f"Extracted {len(tensors)} tensors")
    
    # Save as .pt
    output_path = "rust/model_converted.pt"
    print(f"Saving to {output_path}")
    torch.save(tensors, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
