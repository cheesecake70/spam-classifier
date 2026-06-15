import pickle

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    classifier = load_model()

    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim now!!!",
        "Hey, are we still on for lunch tomorrow?", 
        "URGENT Your bank account has been compromised. Call us immediately.",
        "Can you send me the notes from today's lecture?"
    ]

    for msg in test_messages:
        result = classifier.predict(msg)
        print(f"[{result.upper()}] {msg}")