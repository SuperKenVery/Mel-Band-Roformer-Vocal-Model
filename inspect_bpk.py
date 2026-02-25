
import msgpack
import torch
import numpy as np
import os

import gzip

def inspect_bpk(path):
    print(f"Loading {path}...")
    with open(path, "rb") as f:
        unpacker = msgpack.Unpacker(f)
        for i, obj in enumerate(unpacker):
            print(f"Object {i}: {type(obj)}")
            if isinstance(obj, dict):
                print(f"  Keys: {list(obj.keys())[:5]}")
                # Print one key-value pair info
                k = list(obj.keys())[0]
                v = obj[k]
                # Burnpack format: (id, shape, data) ?
                # Or just tensor data?
                print(f"  Value for {k}: {str(v)[:100]}")
            if i > 2: break
    return


if __name__ == "__main__":
    inspect_bpk("rust/model.bpk")
