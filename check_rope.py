import torch
import rotary_embedding_torch

def main():
    print(dir(rotary_embedding_torch))
    if hasattr(rotary_embedding_torch, 'rotate_half'):
        rotate_half = rotary_embedding_torch.rotate_half
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        print(f"rotate_half([1, 2, 3, 4]) = {rotate_half(x)}")
    else:
        print("rotate_half not found in module.")

if __name__ == "__main__":
    main()
