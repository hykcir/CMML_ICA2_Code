from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd


def evaluate_embeddings(
    X_train_emb,
    y_train,
    X_test_emb,
    y_test,
    model_name,
    classifier="logreg",
    random_state=42,
):
    # Evaluate the quality of the embeddings using a simple classifier 
    # (logistic regression or k-nearest neighbors)
    if classifier == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=4,
            ),
        )

    elif classifier == "knn":
        clf = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                metric="cosine",
            ),
        )

    else:
        raise ValueError("classifier must be 'logreg' or 'knn'")

    # Fit the classifier and evaluate on the test set
    clf.fit(X_train_emb, y_train)
    y_pred = clf.predict(X_test_emb)

    # Compute evaluation metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    result = {
        "model": model_name,
        "classifier": classifier,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "embedding_dim": X_train_emb.shape[1],
    }

    return result, y_pred, clf