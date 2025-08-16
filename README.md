# continuous_deep_embed
Continuous Deep Embed (CDE): codebook‑gated MLPs for Vision Transformers.


## experiment


- `pip install kaggle`
- `kaggle competitions download -c imagenet-object-localization-challenge -p ./data/imagenet`
- `python src/cde_vit.py --bs 1024 --epochs 100 --grid ./grids/full.json --imagenet ./data/imagenet --mlflow --imagenet_layout cls-loc`
