import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the diabetes dataset
diabetes_dataset = pd.read_csv(r'/Users/nikhilsethi/Desktop/EI SYSTEM/diabetes.csv')

# Separate features and labels
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset["Outcome"]

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train the SVM Classifier
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Function to predict diabetes based on user input
def predict_diabetes():
    try:
        # Collect input values from the GUI
        pregnancies = int(entry_pregnancies.get())
        glucose = float(entry_glucose.get())
        blood_pressure = float(entry_bp.get())
        skin_thickness = float(entry_skin.get())
        insulin = float(entry_insulin.get())
        bmi = float(entry_bmi.get())
        dpf = float(entry_dpf.get())
        age = int(entry_age.get())
        
        # Prepare the input data for prediction
        input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        # Standardize the input data
        std_data = scaler.transform(input_data_reshaped)
        
        # Make the prediction
        prediction = classifier.predict(std_data)
        
        # Display the result
        if prediction[0] == 0:
            messagebox.showinfo("Result", "The person is NON-DIABETIC.")
        else:
            messagebox.showinfo("Result", "The person is DIABETIC.")
    
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields.")

# GUI setup using Tkinter
app = tk.Tk()
app.title("Diabetes Prediction")
app.geometry("400x500")

# Labels and entry fields for input
tk.Label(app, text="Diabetes Prediction System", font=("Arial", 16)).pack(pady=10)

tk.Label(app, text="Number of Pregnancies:").pack()
entry_pregnancies = tk.Entry(app)
entry_pregnancies.pack()

tk.Label(app, text="Glucose Level:").pack()
entry_glucose = tk.Entry(app)
entry_glucose.pack()

tk.Label(app, text="Blood Pressure:").pack()
entry_bp = tk.Entry(app)
entry_bp.pack()

tk.Label(app, text="Skin Thickness:").pack()
entry_skin = tk.Entry(app)
entry_skin.pack()

tk.Label(app, text="Insulin Level:").pack()
entry_insulin = tk.Entry(app)
entry_insulin.pack()

tk.Label(app, text="BMI (Body Mass Index):").pack()
entry_bmi = tk.Entry(app)
entry_bmi.pack()

tk.Label(app, text="Diabetes Pedigree Function:").pack()
entry_dpf = tk.Entry(app)
entry_dpf.pack()

tk.Label(app, text="Age:").pack()
entry_age = tk.Entry(app)
entry_age.pack()

# Button to predict
predict_button = tk.Button(app, text="Predict", command=predict_diabetes)
predict_button.pack(pady=20)

# Run the GUI
app.mainloop()
