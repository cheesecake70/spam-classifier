import csv
import random
from classifier import NaiveBayesClassifier
import pickle

def load_data(filepath):
    messages = []
    labels = []
    with open(filepath, encoding="latin-1") as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            label = row[0]   # "spam" or "ham"
            message = row[1] # the text
            labels.append(label)
            messages.append(message)
    return messages, labels

# 80% goes to training, 20% is held back for testing
def split_data(messages, labels, train_ratio=0.8):
    combined = list(zip(messages, labels))
    random.shuffle(combined)
    messages, labels = zip(*combined)
    split = int(len(messages) * train_ratio)
    return (messages[:split], labels[:split],   # train
            messages[split:], labels[split:])   # test

if __name__ == "__main__":
    messages, labels = load_data("spam.csv")
    train_msgs, train_labels, test_msgs, test_labels = split_data(messages, labels)

    classifier = NaiveBayesClassifier()
    classifier.train(train_msgs, train_labels)

    # evaluate accuracy on test set
    correct = 0
    for msg, label in zip(test_msgs, test_labels):
        if classifier.predict(msg) == label:
            correct += 1
    
    accuracy = correct / len(test_msgs) * 100 # how many it got right
    print(f"Accuracy: {accuracy:.2f}%")

    # save the trained model so predict.py can use it
    with open("model.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print("Model saved")