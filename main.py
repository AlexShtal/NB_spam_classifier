from GaussianNB import *

classifier = GaussianNB()
classifier.load("datasets/spam.csv", "windows-1251", text_column_name="v2", spam_column_name="v1")

result = classifier.predict(
    "Hi John, just letting you know I'll be a bit late for dinner tonight. See you soon!")
print(result)

from WindowedClassifier import WindowedClassifier

main_window = WindowedClassifier("datasets/spam.csv", "windows-1251", text_column_name="v2", spam_column_name="v1")
main_window.run()
