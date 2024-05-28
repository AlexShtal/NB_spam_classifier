from tkinter import *
from tkinter import ttk
from GaussianNB import GaussianNB


class WindowedClassifier:
    model = None

    def __init__(self, path: str, encoding: str, text_column_name: str = None, spam_column_name: str = None):
        self.model = GaussianNB()
        self.model.load(path, encoding, text_column_name=text_column_name, spam_column_name=spam_column_name)

    def run(self):
        def validate(text):
            predicted = self.model.predict(text, only_spam_prob=True)
            spam_val.set(int(predicted * 10000) / 100)
            not_spam_val.set(int(10000 - predicted * 10000) / 100)

            spam_val.set(f"{spam_val.get()}%")
            not_spam_val.set(f"{not_spam_val.get()}%")
            return True

        # main window
        main_window = Tk()
        main_window.title("NBSpamClassifier")
        main_window.iconbitmap("icon.ico")
        main_window.geometry('430x130')

        # progress bars
        # spam
        spam_text = Label(text="Spam prob:", font=8)
        spam_text.place(x=15, y=65)

        spam_val = DoubleVar(value="0.0%")
        spam_val_label = Label(textvariable=spam_val, font=8)
        spam_val_label.place(x=350, y=65)

        spam_bar = ttk.Progressbar(main_window, orient="horizontal", length=200, variable=spam_val)
        spam_bar.place(x=140, y=65)

        # not spam
        not_spam_text = Label(text="Not spam prob: ", font=8)
        not_spam_text.place(x=15, y=95)

        not_spam_val = DoubleVar(value="0.0%")
        not_spam_val_label = Label(textvariable=not_spam_val, font=8)
        not_spam_val_label.place(x=350, y=95)

        not_spam_bar = ttk.Progressbar(main_window, orient="horizontal", length=200, variable=not_spam_val)
        not_spam_bar.place(x=140, y=95)

        # text field
        command = (main_window.register(validate), "%P")
        text_field = Entry(main_window, validate="key", validatecommand=command, font=14)
        text_field.place(x=15, y=15, width=400, height=30)

        text_field.focus()

        main_window.mainloop()
