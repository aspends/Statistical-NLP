import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from model import MyModel  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--hidden_size", type=int, default=256, help="The size of the hidden layer.")
    parser.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "tanh", "sigmoid"],
                        help="The nonlinearity function to use.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train for.")
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    data = pd.read_csv(args.featurefile)
    #print(data.iloc[:, 1:].values)
    X = data.iloc[:, 1:-2].values.astype(np.float32)  # features
    y = data.iloc[:, 0].values  # labels
    print(X)
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_train=y_train.astype(np.int32)
    print(type(y_train))
    print(y_train.dtype)
    # Convert the labels to one-hot vectors
    num_classes = len(np.unique(y_train))
    print(num_classes)
    unique_labels=np.unique(y_train)
    print(unique_labels)
    
    all_classes=np.unique(data.iloc[:,-2].values)
    train_classes=np.unique(y_train)
    test_classes=np.unique(y_test)
    
    missing_train_classes = set(all_classes) - set(train_classes)  
    missing_test_classes = set(all_classes) - set(test_classes)  

    print("Missing classes in training data:", missing_train_classes)
    print("Missing classes in testing data:", missing_test_classes)
    
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    # Model creation
    model = MyModel(args.dims, args.hidden_size, num_classes, args.nonlinearity)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training
    print("Training...")
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i in range(len(X_train)):
            inputs = torch.from_numpy(X_train[i]).unsqueeze(0)
            labels = torch.from_numpy(np.array([y_train[i]]))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch {}: Loss = {:.4f}".format(epoch+1, running_loss/len(X_train)))

    # Testing
    print("Testing...")
    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(X_test))
        _, predicted = torch.max(outputs, 1)
        cm = confusion_matrix(y_test, predicted.numpy())
        print("Confusion matrix:\n", cm)
        print("Accuracy: {:.2f}%".format(100 * np.trace(cm) / np.sum(cm)))
        
    print("Done!")
