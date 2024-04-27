from Classifier import GaussianNB

classifier = GaussianNB()
classifier.load("datasets/spam.csv", "windows-1251", "v2", "v1")

result = classifier.predict(
    "Hi John, just letting you know I'll be a bit late for dinner tonight. See you soon!")
print(result)

result = classifier.predict(
    "Congratulations! You've won a free vacation to an exotic destination! Click here to claim your prize now!")
print(result)
