import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib

# Load model and data
model = joblib.load('drug_effectiveness_model.pkl')
df = pd.read_csv("drug_clean.csv")

# Create dictionary for drug data
drug_data = df.groupby("Drug").mean(numeric_only=True).reset_index()

# Initialize GUI window
root = tk.Tk()
root.title("Drug Effectiveness Predictor")
root.geometry("450x400")
root.resizable(False, False)

# Dropdown for drug selection
tk.Label(root, text="Select Drug:", font=("Arial", 12)).pack(pady=5)
drug_var = tk.StringVar()
drug_menu = ttk.Combobox(root, textvariable=drug_var, state="readonly", width=40)
drug_menu['values'] = list(drug_data["Drug"])
drug_menu.pack()

# Labels to display values
info_labels = {}
for label in ["Price", "Reviews", "Satisfaction", "EaseOfUse"]:
    frame = tk.Frame(root)
    frame.pack(pady=3)
    tk.Label(frame, text=f"{label}:", width=15, anchor='w').pack(side='left')
    val_label = tk.Label(frame, text="N/A", width=20, anchor='w', fg='blue')
    val_label.pack(side='left')
    info_labels[label] = val_label

# Function to update displayed values on selection
def update_fields(event=None):
    selected = drug_var.get()
    row = drug_data[drug_data["Drug"] == selected]
    if not row.empty:
        for label in ["Price", "Reviews", "Satisfaction", "EaseOfUse"]:
            val = round(row[label].values[0], 2)
            info_labels[label].config(text=str(val))

# Prediction function
def predict_effectiveness():
    selected = drug_var.get()
    row = drug_data[drug_data["Drug"] == selected]
    if row.empty:
        messagebox.showerror("Error", "Please select a valid drug.")
        return
    features = row[["Price", "Reviews", "Satisfaction", "EaseOfUse"]].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    messagebox.showinfo("Prediction Result", f"Predicted Effectiveness for '{selected}': {prediction:.2f}")

# Button
tk.Button(root, text="Predict Effectiveness", command=predict_effectiveness, bg="green", fg="white", font=("Arial", 11)).pack(pady=20)

# Bind selection event
drug_menu.bind("<<ComboboxSelected>>", update_fields)

root.mainloop()
