# Reviewers' rating prediction based on XLNet

python version=3.8

```bash
! pip install transformers
! pip install sentencepiece
! pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
! pip install tqdm
! pip install optuna
! pip install scikit-learn
```

Run:
```bash
python xlnet.py --experiment_name xlnet
```

# Music Tagging

## Musicnn
Run:
```bash
python muicnn_tuning.py --experiment_name musicnn
```

## Audio Spectrogram Transformer
Run:
```bash
python ast_tuning.py --experiment_name ast
```