# WIND-TURBINE-FAILURE-PREDICTION-PROJECT.
This project uses a Random Forest Classifier to predict wind turbine failures based on a synthetic SCADA dataset. It analyzes features like wind speed, temperature, vibration, and power output to detect potential faults early. The model achieves around 87% accuracy and is integrated with Streamlit for live predictions. 


🌪️ WIND TURBINE FAILURE PREDICTION PROJECT
📘 Overview

This project aims to predict potential wind turbine failures using data collected from SCADA (Supervisory Control and Data Acquisition) systems.
By analyzing real-time operational parameters such as wind speed, temperature, vibration, and power output, the model identifies early warning signs of malfunction or breakdown.

The project uses a Random Forest Classifier trained on a synthetic dataset, achieving approximately 87% accuracy.
A Streamlit web application is also integrated to allow live predictions and interactive analysis.

🧠 Key Features

✅ Predicts potential turbine failures before they occur

🌬️ Uses SCADA parameters (wind speed, temperature, vibration, power output)

🤖 Machine Learning model: Random Forest Classifier

📈 Model accuracy: ~87%

🧮 Includes exploratory data analysis (EDA) and feature importance visualization

💻 Streamlit interface for easy user interaction and live predictions
⚙️ Installation and Setup
1. Clone the repository
git clone https://github.com/yourusername/WIND-TURBINE-FAILURE-PREDICTION-PROJECT.git
cd WIND-TURBINE-FAILURE-PREDICTION-PROJECT

2. Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run the Streamlit app
streamlit run app/streamlit_app.py

📊 Model Details

Algorithm: Random Forest Classifier

Accuracy: ~87% on test data

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Feature Importance:

Vibration

Wind Speed

Temperature

Power Output

🖥️ Streamlit App Overview

The Streamlit dashboard allows users to:

Input custom SCADA readings

View instant predictions (Normal / Failure Risk)

Visualize feature importance and trends

Example:

Wind Speed: 12.4 m/s  
Temperature: 58°C  
Vibration: 0.87 mm/s  
Power Output: 320 kW  
→ Prediction: ⚠️ Failure Risk Detected

🚀 Future Improvements

Incorporate real SCADA data for enhanced realism

Add time-series failure forecasting

Integrate alert system via email/SMS

Experiment with deep learning models (e.g., LSTM for sequential data)

🧩 Tech Stack

Python 3.x

Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Streamlit for UI
📄 License

This project is licensed under the MIT License – see the LICENSE
 file for details.
