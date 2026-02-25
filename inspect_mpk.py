
import gzip
import msgpack
import torch
import numpy as np

def inspect_mpk(path):
    print(f"Loading {path}...")
    try:
        with gzip.open(path, "rb") as f:
            data = msgpack.unpack(f)
    except Exception as e:
        print(f"Gzip failed: {e}")
        with open(path, "rb") as f:
            data = msgpack.unpack(f)
            
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        
        item = data.get("item")
        print(f"Item type: {type(item)}")
        if isinstance(item, dict):
            print(f"Item keys: {list(item.keys())[:5]}")
            # Recursively print structure?
            
        tt = item.get("time_transformers")
        print(f"Time Transformers type: {type(tt)}")
        # If list (Vec<Transformer>)
        if isinstance(tt, list):
            print(f"Time Transformers len: {len(tt)}")
            t0 = tt[0]
            print(f"T0 keys: {t0.keys()}")
            # Check norm
            norm = t0.get("norm")
            print(f"Norm keys: {norm.keys()}")
            gamma = norm.get("gamma")
            print(f"Gamma: {gamma}")


if __name__ == "__main__":
    inspect_mpk("rust/model.mpk")
