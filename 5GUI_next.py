import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load model & dataset
model = joblib.load("drug_effectiveness_model.pkl")
df = pd.read_csv("drug_clean.csv")
drug_data = df.groupby("Drug").mean(numeric_only=True).reset_index()

# -------------------- Main App Window ---------------------
root = tk.Tk()
root.title("💊 Drug Effectiveness Assistant")
root.geometry("750x600")
root.configure(bg="#006400")  # Dark Green Background

# -------------------- Styling -----------------------------
style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background="#006400", foreground="white", font=("Helvetica", 11))
style.configure("TButton", font=("Helvetica", 12), padding=8, background="#2e8b57", foreground="white")
style.map("TButton", background=[("active", "#228b22")])

# -------------------- Title Banner ------------------------
title = tk.Label(root, text="💊 Drug Effectiveness Predictor", font=("Helvetica", 18, "bold"),
                 bg="#2e8b57", fg="white", pady=10)
title.pack(fill="x")

# -------------------- Drug Selector -----------------------
frame_select = tk.Frame(root, bg="#006400")
frame_select.pack(pady=15)

tk.Label(frame_select, text="🔽 Choose a Drug", font=("Helvetica", 13, "bold"),
         bg="#006400", fg="white").pack()
drug_var = tk.StringVar()
drug_menu = ttk.Combobox(frame_select, textvariable=drug_var, width=45, state="readonly")
drug_menu['values'] = list(drug_data["Drug"])
drug_menu.pack(pady=6)

# -------------------- Info Display ------------------------
info_frame = tk.LabelFrame(root, text="📊 Drug Info", bg="#006400", fg="white",
                           font=("Helvetica", 12, "bold"))
info_frame.pack(padx=20, pady=10, fill="x")

info_labels = {}
for item in ["Price", "Reviews", "Satisfaction", "EaseOfUse"]:
    row = tk.Frame(info_frame, bg="#006400")
    row.pack(anchor="w", pady=3)
    tk.Label(row, text=item + ":", font=("Helvetica", 11, "bold"),
             bg="#006400", fg="#ffc107", width=14).pack(side="left")
    info_labels[item] = tk.Label(row, text="--", bg="#006400", fg="white",
                                 font=("Helvetica", 11))
    info_labels[item].pack(side="left")

# -------------------- Update on Selection -----------------
def update_info(event=None):
    drug = drug_var.get()
    row = drug_data[drug_data["Drug"] == drug]
    if not row.empty:
        for key in info_labels:
            value = round(row[key].values[0], 2)
            info_labels[key].config(text=str(value))

drug_menu.bind("<<ComboboxSelected>>", update_info)

# -------------------- Prediction Logic --------------------
def predict():
    drug = drug_var.get()
    row = drug_data[drug_data["Drug"] == drug]
    if row.empty:
        messagebox.showerror("❌ Error", "Please select a valid drug.")
        return
    X = row[["Price", "Reviews", "Satisfaction", "EaseOfUse"]].values.reshape(1, -1)
    prediction = model.predict(X)[0]
    messagebox.showinfo("✅ Prediction", f"Predicted Effectiveness for '{drug}': {prediction:.2f}")
    show_chart(drug, prediction)

# -------------------- Graph Display -----------------------
def show_chart(drug_name, prediction):
    chart_frame = tk.Frame(root, bg="#006400")
    chart_frame.pack(pady=10)
    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
    ax.bar(["Prediction"], [prediction], color="#98fb98")
    ax.set_ylim(0, 5)
    ax.set_title(f"{drug_name} - Effectiveness Score", fontsize=11)
    ax.set_ylabel("Score (0-5)")
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# -------------------- Predict Button ----------------------
predict_btn = ttk.Button(root, text="🔍 Predict Effectiveness", command=predict)
predict_btn.pack(pady=15)

# -------------------- Footer ------------------------------
tk.Label(root, text="Developed by UMIT SAMANTA", bg="#006400", fg="white", font=("Helvetica", 10, "italic")).pack(side="bottom", pady=10)

# -------------------- Start App ---------------------------
root.mainloop()
