import pandas as pd
import re
import math


class GaussianNB:
    word_prob = []

    def load(self, path: str, encoding: str, text_column_name: str, spam_column_name: str):
        dataset = pd.read_csv(path, encoding=encoding)

        df = pd.DataFrame(data={"text": [i for i in dataset[text_column_name]],
                                "spam": [0 if i == "ham" else 1 for i in dataset[spam_column_name]]})

        freq = self.count_words(df)

        all_spam = df['spam'].loc[df['spam'] == 1].count()
        all_not_spam = df['spam'].loc[df['spam'] == 0].count()

        self.word_probs = self.word_probs(freq, all_spam, all_not_spam)

    def predict(self, message):
        message_words = self.tokenize(message)
        spam_prob = not_spam_prob = 0.0

        for word, prob_if_spam, prob_if_not_spam in self.word_probs:
            if word in message_words:
                spam_prob += math.log(prob_if_spam)
                not_spam_prob += math.log(prob_if_not_spam)
            else:
                spam_prob += math.log(1.0 - prob_if_spam)
                not_spam_prob += math.log(1.0 - prob_if_not_spam)

        e_spam_prob = math.exp(spam_prob)
        e_not_spam_prob = math.exp(not_spam_prob)
        prob = e_spam_prob / (e_spam_prob + e_not_spam_prob)

        return round(prob, 5)

    def tokenize(self, message):
        message_lower = message.lower()
        all_words = re.findall("[a-z0-9]+", message_lower)
        return set(all_words)

    def count_words(self, data):
        counts = {}
        for index, row in data.iterrows():
            for word in self.tokenize(row['text']):
                if word not in counts:
                    counts[word] = [0, 0]
                counts[word][0 if row["spam"] else 1] += 1
        return counts

    def word_probs(self, freq, all_spam, all_not_spam, k=1):
        probs = []

        for word, frequency in freq.items():
            probs.append((word, (frequency[0] + k) / (all_spam + 2 * k),
                          (frequency[1] + k) / (all_not_spam + 2 * k)))
        return probs
