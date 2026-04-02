# Hybrid Strategy Results Comparison

| Model | Method | Dataset | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BrainGNN | fmriprep | ds002748 | 0.7238 | 0.8494 | 0.7655 | 0.7923 | 0.7582 |
| BrainGNN | fmriprep | KKI | 0.5243 | 0.1556 | 0.15 | 0.1374 | 0.3908 |
| BrainGNN | fmriprep | NeuroIMAGE | 0.5067 | 0.5276 | 0.68 | 0.581 | 0.468 |
| BrainGNN | fmriprep | OHSU | 0.4958 | 0.4444 | 0.3536 | 0.377 | 0.5077 |
| BrainGNN | deepprep | ds002748 | 0.6267 | 0.8828 | 0.5873 | 0.6781 | 0.6555 |
| BrainGNN | deepprep | KKI | 0.4725 | 0.1744 | 0.4333 | 0.2425 | 0.5789 |
| BrainGNN | deepprep | NeuroIMAGE | 0.5356 | 0.3571 | 0.4 | 0.3571 | 0.658 |
| BrainGNN | deepprep | OHSU | 0.5333 | 0.3911 | 0.3893 | 0.386 | 0.5611 |
| BrainNetCNN | fmriprep | ds002748 | 0.7095 | 0.7853 | 0.8436 | 0.8059 | 0.6716 |
| BrainNetCNN | fmriprep | KKI | 0.6963 | 0.2 | 0.13 | 0.1556 | 0.5258 |
| BrainNetCNN | fmriprep | NeuroIMAGE | 0.5089 | 0.44 | 0.4 | 0.4062 | 0.514 |
| BrainNetCNN | fmriprep | OHSU | 0.56 | 0.511 | 0.5036 | 0.4868 | 0.5222 |
| BrainNetCNN | deepprep | ds002748 | 0.6143 | 0.6729 | 0.8327 | 0.7369 | 0.5211 |
| BrainNetCNN | deepprep | KKI | 0.6317 | 0.0571 | 0.1 | 0.0727 | 0.5106 |
| BrainNetCNN | deepprep | NeuroIMAGE | 0.6667 | 0.7321 | 0.72 | 0.6538 | 0.788 |
| BrainNetCNN | deepprep | OHSU | 0.4675 | 0.4196 | 0.45 | 0.4078 | 0.4939 |
