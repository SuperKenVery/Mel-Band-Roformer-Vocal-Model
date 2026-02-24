# Python
pixi run python inference.py --config_path configs/config_vocals_mel_band_roformer.yaml --model_path /Users/bytedance/Downloads/MelBandRoformer.ckpt --input_folder /Users/bytedance/Desktop/music-instrumental-extract/inputs --store_dir /Users/bytedance/Desktop/music-instrumental-extract/outputs

# Rust
cd rust/
cargo run -- infer \
      -m ./model.bpk \
      -i ~/Desktop/music-instrumental-extract/inputs/syws.wav \
      -o ~/Desktop/music-instrumental-extract/outputs \
      -c ../configs/config_vocals_mel_band_roformer.yaml \
      -d gpu
