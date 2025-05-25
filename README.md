# Drug-Recommendation-System
A data science project that analyzes patient drug reviews to recommend the most suitable medications for specific medical conditions. This end-to-end system combines data analysis, machine learning, visualization, and interactive interfaces built with Python.

🚀 Features:-
📊 Analyzes drug effectiveness based on satisfaction, ease of use, and price.
🧠 Machine learning model (Random Forest) to predict drug effectiveness.
🧮 Composite score-based recommendation engine.
💡 Visual insights with bar and scatter plots.
🖥️ GUI with Tkinter and optional Streamlit interface.
🌐 API-ready backend with FastAPI (coming soon).

📁 Project Structure
Drug-Recommendation-System/
│
├── 1main.py                # Composite score-based recommender
├── 2main_set.py            # Improved version with better output
├── 3ml_model.py            # ML model for effectiveness prediction
├── 4GUI.py                 # Tkinter GUI application
├── 5GUI_next.py            # Advanced GUI (in progress)
├── 7API_Frontend.py        # FastAPI and UI integration (coming soon)
├── API_Streamlit6.py       # Streamlit-based frontend (optional)
│
├── drug_clean.csv          # Cleaned drug dataset
├── drug_effectiveness_model.pkl  # Trained ML model
│
├── Bar_Graph_Effect_Prediction.png
├── Scatter_Graph_Drug_Effect.png
└── Doc_drug_data.docx      # Supporting documentation


🧠 How It Works
Data Cleaning & Feature Engineering:
Removes missing values and calculates a composite score using satisfaction, ease of use, and effectiveness.

Drug Recommendation Logic:
Filters by condition and ranks drugs using the composite score.

Machine Learning Model:

Trains a RandomForestRegressor to predict drug effectiveness.

Visualizes predictions vs. actual values and feature importance.

GUI Prediction:

User selects a drug from a dropdown.

The model predicts its effectiveness instantly.

🔧 Tech Stack
Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
Tkinter (Desktop GUI)
Joblib (Model persistence)
FastAPI + Streamlit (Planned web APIs/UI)
VS Code (Development Environment)

📄 License
This project is licensed under the MIT License.

