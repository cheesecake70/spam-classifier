import math

class NaiveBayesClassifier:
    def __init__(self):
        # number of times each word appears in spam/ham
        self.word_counts = {"spam":{}, "ham":{}} 
        # total number of spam/ham messages seen
        self.class_counts = {"spam":0, "ham":0}
        # all unique words
        self.vocab = set()

    def tokenize(self, text):
        # converts message into a list of words
        return text.lower().split()
    
    def train(self, messages, labels):
        for message, label in zip(messages, labels):
            self.class_counts[label] += 1 # increments counter of spam/ham
            for word in self.tokenize(message):
                # adds individual words to vocab + spam/ham w count
                self.vocab.add(word)
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0
                self.word_counts[label][word] += 1

    def predict(self, message):
        # message is the new message to be classified

        total_messages = sum(self.class_counts.values())
        vocab_size = len(self.vocab)
    
        log_probs = {}
        for label in ["spam", "ham"]:
            # probs of spam and ham in entire dataset
            log_probs[label] = math.log(self.class_counts[label]/total_messages)
            # no of words across all spam n ham
            total_words_in_class = sum(self.word_counts[label].values())

            # loop through words in message that we are classifying
            for word in self.tokenize(message):
                if word not in self.vocab:
                    continue  # skip completely unknown words
                
                # Laplace smoothing: (count + 1) / (total_words_in_class + vocab_size)
                # we added 1 to the word we are looking at's count
                # by dividing by total_words_in_class + vocab_size, 
                # we are pretending we did that for every word in the vocabulary
                word_count = self.word_counts[label].get(word, 0)
                word_prob = (word_count + 1) / (total_words_in_class + vocab_size)
                log_probs[label] += math.log(word_prob)

        # using log to avoid the v small values to multiply n give zero

        return "spam" if log_probs["spam"] > log_probs["ham"] else "ham"