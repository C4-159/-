import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os
import sqlite3
from datetime import datetime

conn = sqlite3.connect("history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment TEXT,
    risk TEXT,
    time TEXT
)
""")

conn.commit()

def save_to_db(text, sentiment, risk):
    cursor.execute("""
    INSERT INTO history (text, sentiment, risk, time)
    VALUES (?, ?, ?, ?)
    """, (
        text,
        sentiment,
        risk,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
nltk.download('stopwords')

# -----------------------------
# Дані та покращена очистка
# -----------------------------
df = pd.read_csv("dataset.csv").dropna(subset=["Sentence", "Sentiment"])

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    # видаляємо пунктуацію та цифри
    text = "".join([c for c in text if c not in string.punctuation and not c.isdigit()])
    words = text.split()
    # видаляємо стоп-слова та дуже короткі слова
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return " ".join(words)

df["clean_news"] = df["Sentence"].apply(preprocess)

# -----------------------------
# TF-IDF та покращена модель
# -----------------------------
X = df["clean_news"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

if os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl"):

    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    print("MODEL LOADED!")

else:

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=5000
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("MODEL TRAINED!")

    print(classification_report(y_test, predictions))

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("MODEL TRAINED! METRICS:")
    print(classification_report(y_test, predictions))
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

# -----------------------------
# Risk Event
# -----------------------------
risk_dict = {
    "bankruptcy": 3,
    "crisis": 2,
    "loss": 2,
    "losses": 2,
    "debt": 2,
    "inflation": 2,
    "decline": 2,
    "default": 3,
    "sanctions": 2,
    "war": 3,
    "instability": 2
}
def detect_risk(news):
    clean = preprocess(news)
    sentiment_pred = model.predict(vectorizer.transform([clean]))[0]

    words = clean.split()
    score = sum(risk_dict.get(w, 0) for w in words)

    if sentiment_pred.lower() == "negative":
        score += 1

    risk_flag = "RISK EVENT" if score >= 2 else "NO RISK"

    return sentiment_pred, risk_flag

# -----------------------------
# GUI (DARK FINTECH VERSION)
# -----------------------------

window = tk.Tk()
window.title("Financial Risk Detection System")
window.geometry("900x700")

# -------- COLORS --------
BG = "#121826"
PANEL = "#1e2433"
CARD = "#262d40"
TEXT = "#e6e6e6"
ACCENT = "#4ea1ff"
DANGER = "#ff4d4d"
SUCCESS = "#3ddc84"

window.configure(bg=BG)

# -------- STYLE --------
style = ttk.Style()
style.theme_use("clam")

style.configure("TFrame", background=BG)
style.configure("TLabel", background=BG, foreground=TEXT, font=("Arial", 11))

style.configure("Header.TLabel",
                background=BG,
                foreground=ACCENT,
                font=("Arial", 18, "bold"))

style.configure("TButton",
                font=("Arial", 11),
                padding=6,
                background=PANEL,
                foreground=TEXT)

style.map("TButton",
          background=[("active", "#2f3a55")])

style.configure("TNotebook", background=BG, borderwidth=0)
style.configure("TNotebook.Tab",
                background=PANEL,
                foreground=TEXT,
                padding=10)

style.map("TNotebook.Tab",
          background=[("selected", CARD)],
          foreground=[("selected", ACCENT)])

# -------- HEADER --------
header = ttk.Frame(window)
header.pack(fill="x", pady=10)

ttk.Label(header,
          text="Financial Risk Detection System",
          style="Header.TLabel").pack()

# -------- NOTEBOOK --------
tab_control = ttk.Notebook(window)
tab_control.pack(expand=1, fill="both", padx=10, pady=10)

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)

tab_control.add(tab1, text="Перевірка новини")
tab_control.add(tab2, text="Статистика")
tab_control.add(tab3, text="Графіки")
tab_control.add(tab4, text="Тестування")

# =============================
# TAB 1
# =============================
frame1 = ttk.Frame(tab1, padding=15)
frame1.pack(fill="both", expand=True)

ttk.Label(frame1,
          text="Введіть фінансову новину:",
          font=("Arial", 12, "bold")).pack(anchor="w")

news_input = scrolledtext.ScrolledText(
    frame1,
    height=6,
    font=("Segoe UI", 11),
    bg=CARD,
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat"
)
news_input.pack(fill="x", pady=10)

btn_row = ttk.Frame(frame1)
btn_row.pack(fill="x", pady=5)

def clear_text():
    news_input.delete("1.0", tk.END)
    result_text.config(state='normal')
    result_text.delete("1.0", tk.END)
    result_text.config(state='disabled')

def check_news():
    news = news_input.get("1.0", tk.END).strip()
    
    if not news:
        messagebox.showwarning("Увага", "Введіть текст новини!")
        return

    sentiment, risk = detect_risk(news)

    # 💾 ОЦЕ ТУТ правильно
    save_to_db(news, sentiment, risk)

    result_text.config(state='normal')
    result_text.delete("1.0", tk.END)

    result_text.insert(tk.END, "РЕЗУЛЬТАТ АНАЛІЗУ\n", "title")
    result_text.insert(tk.END, "-"*40 + "\n\n")

    result_text.insert(tk.END, f"{news}\n\n")
    result_text.insert(tk.END, f"Тональність: {sentiment}\n")

    if risk == "RISK EVENT":
        result_text.insert(tk.END, f"Ризик: {risk}\n", "risk")
    else:
        result_text.insert(tk.END, f"Ризик: {risk}\n", "safe")

    result_text.config(state='disabled')

ttk.Button(btn_row, text="Перевірити", command=check_news).pack(side="left", padx=5)
ttk.Button(btn_row, text="Очистити", command=clear_text).pack(side="left", padx=5)

result_text = scrolledtext.ScrolledText(
    frame1,
    height=12,
    font=("Consolas", 10),
    bg=PANEL,
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat"
)
result_text.pack(fill="both", expand=True, pady=10)

result_text.tag_config("risk", foreground=DANGER, font=("Arial", 11, "bold"))
result_text.tag_config("safe", foreground=SUCCESS, font=("Arial", 11, "bold"))
result_text.tag_config("title", foreground=ACCENT, font=("Arial", 12, "bold"))

result_text.config(state='disabled')

def show_history():
    stats_text.config(state='normal')
    stats_text.delete("1.0", tk.END)

    cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()

    stats_text.insert(tk.END, "ІСТОРІЯ АНАЛІЗІВ\n\n")

    for r in rows:
        stats_text.insert(
            tk.END,
            f"{r[4]} | {r[1][:60]}...\n"
            f"Sentiment: {r[2]} | Risk: {r[3]}\n\n"
        )

    stats_text.config(state='disabled')
# =============================
# TAB 2
# =============================
frame2 = ttk.Frame(tab2, padding=15)
frame2.pack(fill="both", expand=True)

stats_text = scrolledtext.ScrolledText(
    frame2,
    font=("Consolas", 10),
    bg=PANEL,
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat"
)
stats_text.pack(fill="both", expand=True)

def show_statistics():
    stats_text.config(state='normal')
    stats_text.delete("1.0", tk.END)

    stats_text.insert(tk.END, "СТАТИСТИКА ДАНИХ\n\n")
    stats_text.insert(tk.END, df["Sentiment"].value_counts().to_string())

    results = []
    for _, row in df.iterrows():
        sentiment, risk = detect_risk(row["Sentence"])
        results.append({
            "Sentence": row["Sentence"],
            "Sentiment": sentiment,
            "Risk": risk
        })

    pd.DataFrame(results).to_csv("risk_results.csv", index=False)

    stats_text.insert(tk.END, "\n\n✔ risk_results.csv створено")

    stats_text.config(state='disabled')

ttk.Button(frame2, text="Оновити статистику",
           command=show_statistics).pack(pady=10)

ttk.Button(
    frame2,
    text="Показати історію",
    command=show_history
).pack(pady=5)

# =============================
# TAB 3
# =============================
# =============================
# TAB 3 - PLOTLY CHARTS
# =============================
frame3 = ttk.Frame(tab3, padding=15)
frame3.pack(fill="both", expand=True)

import plotly.express as px

def show_charts():
    for w in frame3.winfo_children():
        w.destroy()

    df["Risk"] = df["Sentence"].apply(lambda x: detect_risk(x)[1])

    fig1 = px.histogram(
        df,
        x="Sentiment",
        color="Sentiment",
        title="Розподіл тональності"
    )
    fig1.show()

    fig2 = px.histogram(
        df,
        x="Risk",
        color="Risk",
        title="Risk Events"
    )
    fig2.show()

    ttk.Label(
        frame3,
        text="✔ Графіки відкрито у браузері (Plotly)",
        foreground=ACCENT
    ).pack(pady=20)

ttk.Button(
    frame3,
    text="Показати інтерактивні графіки",
    command=show_charts
).pack(pady=10)
# =============================
# TAB 4 - MODEL TESTS
# =============================

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

frame4 = ttk.Frame(tab4, padding=15)
frame4.pack(fill="both", expand=True)

test_text = scrolledtext.ScrolledText(
    frame4,
    font=("Consolas", 10),
    bg=PANEL,
    fg=TEXT,
    insertbackground=TEXT,
    relief="flat"
)

test_text.pack(fill="both", expand=True)


def run_tests():

    test_text.config(state='normal')
    test_text.delete("1.0", tk.END)

    # =============================
    # TEST 1 - DATASET INFO
    # =============================

    test_text.insert(tk.END, "TEST 1 - DATASET INFO\n", "title")
    test_text.insert(tk.END, "-" * 60 + "\n")

    test_text.insert(
        tk.END,
        f"Total records: {len(df)}\n\n"
    )

    test_text.insert(
        tk.END,
        str(df["Sentiment"].value_counts())
    )

    # =============================
    # TEST 2 - TRAIN / TEST SPLIT
    # =============================

    test_text.insert(
        tk.END,
        "\n\nTEST 2 - TRAIN / TEST SPLIT\n",
        "title"
    )

    test_text.insert(tk.END, "-" * 60 + "\n")

    test_text.insert(
        tk.END,
        f"Train size: {X_train.shape[0]}\n"
    )

    test_text.insert(
        tk.END,
        f"Test size: {X_test.shape[0]}\n"
    )

    # =============================
    # TEST 3 - ACCURACY
    # =============================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    test_text.insert(
        tk.END,
        "\n\nTEST 3 - ACCURACY\n",
        "title"
    )

    test_text.insert(tk.END, "-" * 60 + "\n")

    test_text.insert(
        tk.END,
        f"Accuracy: {accuracy:.4f}\n"
    )

    # =============================
    # TEST 4 - CLASSIFICATION REPORT
    # =============================

    report = classification_report(y_test, predictions)

    test_text.insert(
        tk.END,
        "\n\nTEST 4 - CLASSIFICATION REPORT\n",
        "title"
    )

    test_text.insert(tk.END, "-" * 60 + "\n")

    test_text.insert(tk.END, report)

    # =============================
    # TEST 5 - CONFUSION MATRIX
    # =============================

    cm = confusion_matrix(y_test, predictions)

    test_text.insert(
        tk.END,
        "\n\nTEST 5 - CONFUSION MATRIX\n",
        "title"
    )

    test_text.insert(tk.END, "-" * 60 + "\n")

    test_text.insert(tk.END, str(cm))

    # =============================
    # SAMPLE NEWS TEST
    # =============================

    sample_news = "Company profits increased significantly this quarter"

    sample_clean = preprocess(sample_news)

    prediction = model.predict(
        vectorizer.transform([sample_clean])
    )[0]

    test_text.insert(
        tk.END,
        "\n\nSAMPLE NEWS TEST\n",
        "title"
    )

    test_text.insert(tk.END, "-" * 60 + "\n")

    test_text.insert(
        tk.END,
        f"News: {sample_news}\n"
    )

    test_text.insert(
        tk.END,
        f"Prediction: {prediction}\n"
    )

    test_text.config(state='disabled')


test_text.tag_config(
    "title",
    foreground=ACCENT,
    font=("Arial", 12, "bold")
)

ttk.Button(
    frame4,
    text="Запустити тести",
    command=run_tests
).pack(pady=10)
# =============================
# STATUS BAR
# =============================
status = ttk.Label(
    window,
    text="Модель готова до роботи",
    background=PANEL,
    foreground=TEXT,
    anchor="w",
    padding=5
)
status.pack(side="bottom", fill="x")

window.mainloop()