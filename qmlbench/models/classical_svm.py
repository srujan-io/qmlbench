from sklearn.svm import SVC
#scikit implementation of SVM classifier
def train_classical_svm(
    X_train,
    y_train,
    C=1.0,
    gamma="scale"
): 
    #kernel - rbf -> use RBF similarity function
    #C=1.0 -> controls how strongly the SVM penalizes training mistakes, larger values -> punish mistakes more
    #gamma = scale -> tells scikit learn to automatically choose gamma based on training data's feature count and variance
    
    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma
    )

    model.fit(X_train, y_train)

    return model
