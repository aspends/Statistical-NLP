import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

def read_documents(inputdir):
    texts = []
    labels = []
    for author in os.listdir(inputdir):
        author_dir = os.path.join(inputdir, author)
        if os.path.isdir(author_dir):
            for filename in os.listdir(author_dir):
                filepath = os.path.join(author_dir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        # Strip email headers and signatures
                        text = re.sub(r"^.*\nFrom:.*\n", "", text, flags=re.MULTILINE)
                        text = re.sub(r"\n-{2,}.*\n.*$", "", text, flags=re.MULTILINE)
                        texts.append(text)
                        labels.append(author)
    return texts, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default=20, help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    texts, labels = read_documents(args.inputdir)

    print("Processing {} documents...".format(len(texts)))
    vectorizer = CountVectorizer(stop_words="english", max_features=10000)
    X = vectorizer.fit_transform(texts)

    print("Reducing dimensionality to {}...".format(args.dims))
    svd = TruncatedSVD(n_components=args.dims, random_state=42)
    X_reduced = svd.fit_transform(X)

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels, test_size=args.testsize/100, random_state=42)
    train_data = pd.DataFrame(X_train)
    train_data["label"] = y_train
    train_data["set"] = "train"
    test_data = pd.DataFrame(X_test)
    test_data["label"] = y_test
    test_data["set"] = "test"
    all_data = pd.concat([train_data, test_data])

    print("Writing to {}...".format(args.outputfile))
    all_data.to_csv(args.outputfile, index=False)

    print("Done!")

