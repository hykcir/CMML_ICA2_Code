# CMML_ICA2_Code

All codes used for analysis and plotting are shown below

1. `emb_eva.py`: The utility script containing the `evaluate_embeddings` function which standardizes embedding evaluation across different models using Logistic Regression or K-Nearest Neighbors (KNN) classifiers, returning metrics like Accuracy, F1, Precision and Recall

2. `scGPT.ipynb`: Extracts and evaluates cell embeddings from a pretrained scGPT model

3. `geneformer.ipynb`: Extracts and evaluates cell embeddings from a pretrained geneformer model

4. `UCE.ipynb`:Extracts and evaluates cell embeddings from a pretrained UCE model
  
5. `scANVI.ipynb`: Trains a scVI and its semi-supervised extension scANVI to predict cell types annotation, and evaluates model's performance

6. `PCA_KNN.ipynb`: Establishes and evaluates a baseline model for cell type annotation by applying KNN classifier on PCA embeddings

7. `PCA_LogReg.ipynb`: Establishes and evaluates a baseline model for cell type annotation by applying logistic regression classifier on PCA embeddings
