# Total Metrics Comparison (Hybrid NNs + SVM/RF)

| Framework | Dataset | Model | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cpac-default | ds002748 | Logistic Regression_L1 | 0.5644 | 0.7429 | 0.619 | 0.6721 | 0.5798 |
| cpac-default | ds002748 | Logistic Regression_PCA | 0.5444 | 0.725 | 0.6429 | 0.6667 | 0.5084 |
| cpac-default | ds002748 | Random Forest_L1 | 0.6022 | 0.7195 | 0.7333 | 0.7214 | 0.5935 |
| cpac-default | ds002748 | Random Forest_PCA | 0.56 | 0.6684 | 0.7286 | 0.6936 | 0.5294 |
| cpac-default | ds002748 | SVM_L1 | 0.6244 | 0.7362 | 0.7333 | 0.7314 | 0.5756 |
| cpac-default | ds002748 | SVM_PCA | 0.7311 | 0.7984 | 0.8524 | 0.8197 | 0.5693 |
| cpac-llm | ds002748 | Logistic Regression_L1 | 0.6472 | 0.8006 | 0.7 | 0.7241 | 0.5824 |
| cpac-llm | ds002748 | Logistic Regression_PCA | 0.6 | 0.7548 | 0.6762 | 0.7048 | 0.5938 |
| cpac-llm | ds002748 | Random Forest_L1 | 0.575 | 0.7113 | 0.7048 | 0.6986 | 0.4986 |
| cpac-llm | ds002748 | Random Forest_PCA | 0.6722 | 0.744 | 0.8429 | 0.7881 | 0.5128 |
| cpac-llm | ds002748 | SVM_L1 | 0.6444 | 0.7556 | 0.7667 | 0.7445 | 0.4347 |
| cpac-llm | ds002748 | SVM_PCA | 0.6972 | 0.7798 | 0.8048 | 0.7846 | 0.5739 |
| deepprep | KKI | BrainGNN | 0.4725 | 0.1744 | 0.4333 | 0.2425 | 0.5789 |
| deepprep | KKI | BrainNetCNN | 0.6317 | 0.0571 | 0.1 | 0.0727 | 0.5106 |
| deepprep | KKI | Linear SVM_Robust | 0.6583 | 0.2833 | 0.3167 | 0.2967 | 0.0 |
| deepprep | KKI | Ridge_Robust | 0.6317 | 0.1667 | 0.2 | 0.18 | 0.0 |
| deepprep | NeuroIMAGE | BrainGNN | 0.5356 | 0.3571 | 0.4 | 0.3571 | 0.658 |
| deepprep | NeuroIMAGE | BrainNetCNN | 0.6667 | 0.7321 | 0.72 | 0.6538 | 0.788 |
| deepprep | NeuroIMAGE | Linear SVM_Robust | 0.6667 | 0.6833 | 0.68 | 0.6697 | 0.0 |
| deepprep | NeuroIMAGE | Ridge_Robust | 0.7089 | 0.79 | 0.64 | 0.6876 | 0.0 |
| deepprep | OHSU | BrainGNN | 0.5333 | 0.3911 | 0.3893 | 0.386 | 0.5611 |
| deepprep | OHSU | BrainNetCNN | 0.4675 | 0.4196 | 0.45 | 0.4078 | 0.4939 |
| deepprep | OHSU | Linear SVM_Robust | 0.4567 | 0.33 | 0.3071 | 0.3167 | 0.0 |
| deepprep | OHSU | Ridge_Robust | 0.5083 | 0.4962 | 0.3357 | 0.3918 | 0.0 |
| deepprep | ds002748 | BrainGNN | 0.6267 | 0.8828 | 0.5873 | 0.6781 | 0.6555 |
| deepprep | ds002748 | BrainNetCNN | 0.6143 | 0.6729 | 0.8327 | 0.7369 | 0.5211 |
| deepprep | ds002748 | Linear SVM_Robust | 0.5962 | 0.6922 | 0.7873 | 0.7291 | 0.0 |
| deepprep | ds002748 | Ridge_Robust | 0.541 | 0.6714 | 0.7073 | 0.684 | 0.0 |
| fmriprep | KKI | BrainGNN | 0.5243 | 0.1556 | 0.15 | 0.1374 | 0.3908 |
| fmriprep | KKI | BrainNetCNN | 0.6963 | 0.2 | 0.13 | 0.1556 | 0.5258 |
| fmriprep | KKI | Linear SVM_Robust | 0.7449 | 0.6 | 0.27 | 0.3633 | 0.0 |
| fmriprep | KKI | Ridge_Robust | 0.7684 | 0.7 | 0.35 | 0.4367 | 0.0 |
| fmriprep | NeuroIMAGE | BrainGNN | 0.5067 | 0.5276 | 0.68 | 0.581 | 0.468 |
| fmriprep | NeuroIMAGE | BrainNetCNN | 0.5089 | 0.44 | 0.4 | 0.4062 | 0.514 |
| fmriprep | NeuroIMAGE | Linear SVM_Robust | 0.5933 | 0.6243 | 0.64 | 0.6267 | 0.0 |
| fmriprep | NeuroIMAGE | Ridge_Robust | 0.5733 | 0.68 | 0.48 | 0.5397 | 0.0 |
| fmriprep | OHSU | BrainGNN | 0.4958 | 0.4444 | 0.3536 | 0.377 | 0.5077 |
| fmriprep | OHSU | BrainNetCNN | 0.56 | 0.511 | 0.5036 | 0.4868 | 0.5222 |
| fmriprep | OHSU | Linear SVM_Robust | 0.535 | 0.5057 | 0.475 | 0.4856 | 0.0 |
| fmriprep | OHSU | Ridge_Robust | 0.5333 | 0.5146 | 0.4714 | 0.4857 | 0.0 |
| fmriprep | ds002748 | BrainGNN | 0.7238 | 0.8494 | 0.7655 | 0.7923 | 0.7582 |
| fmriprep | ds002748 | BrainNetCNN | 0.7095 | 0.7853 | 0.8436 | 0.8059 | 0.6716 |
| fmriprep | ds002748 | Random Forest_Optimized | 0.6686 | 0.7011 | 0.9218 | 0.7962 | 0.493 |
| fmriprep | ds002748 | SVM_Optimized | 0.641 | 0.7227 | 0.8073 | 0.7569 | 0.5537 |
