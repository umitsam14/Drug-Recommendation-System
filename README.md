# 💊 Drug Recommendation System

An end-to-end data science project that analyzes patient drug reviews to recommend the most suitable medications for specific medical conditions. This system combines data preprocessing, machine learning, visual analytics, and interactive interfaces, all built using Python.

---

## 🚀 Key Features

- **📊 Drug Effectiveness Analysis** — Evaluates drugs based on user satisfaction, ease of use, and price.
- **🧠 Machine Learning Model** — Uses Random Forest Regressor to predict drug effectiveness.
- **🧮 Recommendation Engine** — Ranks drugs using a composite scoring algorithm.
- **📈 Visual Insights** — Bar and scatter plots visualize model performance and key metrics.
- **🖥️ Interactive GUI** — Built with Tkinter (desktop) and optionally Streamlit (web).
- **🌐 API-Ready Backend** — FastAPI integration in progress for web deployment.

---

## 📁 Project Structure

<pre>Drug-Recommendation-System/
│
├── 1main.py                # Composite score-based recommender
├── 2main_set.py            # Improved version with better output
├── 3ml_model.py            # ML model for effectiveness prediction
├── 4GUI.py                 # Tkinter GUI application
├── 5GUI_next.py            # Advanced GUI (in progress)
├── 7API_Frontend.py        # FastAPI and UI integration (coming soon)
├── API_Streamlit6.py       # Streamlit-based frontend (optional)
├── drug_clean.csv          # Cleaned drug dataset
├── drug_effectiveness_model.pkl  # Trained ML model
├── Bar_Graph_Effect_Prediction.png
├── Scatter_Graph_Drug_Effect.png
└── Doc_drug_data.docx      # Supporting documentation</pre>


---

## 🧠 How It Works

### 1. Data Cleaning & Feature Engineering
- Handles missing values.
- Computes a composite score from satisfaction, ease of use, and effectiveness.

### 2. Drug Recommendation Logic
- Filters drugs by condition.
- Ranks them based on the composite score.

### 3. Machine Learning Prediction
- Trains a `RandomForestRegressor` model to predict effectiveness.
- Visualizes predictions vs. actual values and feature importance.

### 4. GUI Prediction
- User selects a drug from a dropdown.
- The model instantly predicts its effectiveness.

---

## 🔧 Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Desktop UI:** Tkinter
- **Web UI (optional):** Streamlit
- **Backend (upcoming):** FastAPI
- **IDE:** Visual Studio Code

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
