import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# ГОЛОВНЕ ВІКНО
# =========================

root = tk.Tk()
root.title("Financial Sentiment Analysis System")
root.geometry("1100x700")
root.configure(bg="#1e1e1e")

# =========================
# СТИЛІ
# =========================

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "TButton",
    font=("Arial", 11),
    padding=8
)

style.configure(
    "TLabel",
    background="#1e1e1e",
    foreground="white",
    font=("Arial", 12)
)

# =========================
# ЗАГОЛОВОК
# =========================

title_label = tk.Label(
    root,
    text="Financial Sentiment Analysis System",
    font=("Arial", 20, "bold"),
    bg="#1e1e1e",
    fg="#00ffcc"
)

title_label.pack(pady=10)

# =========================
# ТЕКСТОВЕ ПОЛЕ
# =========================

text_area = ScrolledText(
    root,
    width=130,
    height=35,
    bg="#2b2b2b",
    fg="white",
    insertbackground="white",
    font=("Consolas", 10)
)

text_area.pack(padx=10, pady=10)

# =========================
# ФУНКЦІЯ АНАЛІЗУ
# =========================

def run_analysis():

    text_area.delete(1.0, tk.END)

    try:
        # =========================
        # ЗАВАНТАЖЕННЯ ДАТАСЕТУ
        # =========================

        df = pd.read_csv(
            r"D:\навчання\практика\dataset.csv"
        )

        text_area.insert(
            tk.END,
            "DATASET SUCCESSFULLY LOADED\n\n"
        )

        # =========================
        # ПЕРШІ РЯДКИ
        # =========================

        text_area.insert(
            tk.END,
            "FIRST 5 ROWS:\n\n"
        )

        text_area.insert(
            tk.END,
            str(df.head())
        )

        text_area.insert(tk.END, "\n\n")

        # =========================
        # РОЗПОДІЛ КЛАСІВ
        # =========================

        text_area.insert(
            tk.END,
            "CLASS DISTRIBUTION:\n\n"
        )

        text_area.insert(
            tk.END,
            str(df["Sentiment"].value_counts())
        )

        text_area.insert(tk.END, "\n\n")

        # =========================
        # ФОРМУВАННЯ X та y
        # =========================

        X = df["Sentence"]
        y = df["Sentiment"]

        # =========================
        # TRAIN / TEST SPLIT
        # =========================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        text_area.insert(
            tk.END,
            "TRAIN / TEST SPLIT:\n\n"
        )

        text_area.insert(
            tk.END,
            f"Train size: {len(X_train)}\n"
        )

        text_area.insert(
            tk.END,
            f"Test size: {len(X_test)}\n\n"
        )

        # =========================
        # TF-IDF ВЕКТОРИЗАЦІЯ
        # =========================

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        text_area.insert(
            tk.END,
            "TF-IDF VECTORIZATION COMPLETED\n\n"
        )

        # =========================
        # НАВЧАННЯ МОДЕЛІ
        # =========================

        model = LogisticRegression(max_iter=1000)

        model.fit(X_train_tfidf, y_train)

        text_area.insert(
            tk.END,
            "MODEL TRAINED SUCCESSFULLY\n\n"
        )

        # =========================
        # ПРОГНОЗ
        # =========================

        y_pred = model.predict(X_test_tfidf)

        # =========================
        # ACCURACY
        # =========================

        accuracy = accuracy_score(y_test, y_pred)

        text_area.insert(
            tk.END,
            f"ACCURACY: {accuracy:.4f}\n\n"
        )

        # =========================
        # CLASSIFICATION REPORT
        # =========================

        report = classification_report(y_test, y_pred)

        text_area.insert(
            tk.END,
            "CLASSIFICATION REPORT:\n\n"
        )

        text_area.insert(
            tk.END,
            report
        )

        text_area.insert(tk.END, "\n")

        # =========================
        # CONFUSION MATRIX
        # =========================

        cm = confusion_matrix(y_test, y_pred)

        text_area.insert(
            tk.END,
            "CONFUSION MATRIX:\n\n"
        )

        text_area.insert(
            tk.END,
            str(cm)
        )

        text_area.insert(tk.END, "\n\n")

        # =========================
        # ТЕСТ НОВИНИ
        # =========================

        sample_news = [
            "Company profits increased significantly this quarter"
        ]

        sample_vector = vectorizer.transform(sample_news)

        prediction = model.predict(sample_vector)

        text_area.insert(
            tk.END,
            "SAMPLE NEWS TEST:\n\n"
        )

        text_area.insert(
            tk.END,
            f"News: {sample_news[0]}\n"
        )

        text_area.insert(
            tk.END,
            f"Prediction: {prediction[0]}\n"
        )

    except Exception as e:

        text_area.insert(
            tk.END,
            f"ERROR:\n{str(e)}"
        )

# =========================
# КНОПКА ЗАПУСКУ
# =========================

run_button = ttk.Button(
    root,
    text="Run Financial Analysis",
    command=run_analysis
)

run_button.pack(pady=10)

# =========================
# ЗАПУСК
# =========================

root.mainloop()