import gradio as gr
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_model(fp):
    "Load the model to reuse in App"
    with open(fp, "rb") as f:
        model = pickle.load(f)
    return model

best_model = load_model("Src/churn_model.pkl")

# Creating the preprocessing pipeline
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']

numerical_features = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']

categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

def predict(SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
            MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure):

    data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
             MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_df['tenure_group'] = pd.cut(new_df['tenure'], range(1, 80, 12), right=False, labels=labels)
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    processed_data = preprocessor.transform(new_df)
    single = best_model.predict(processed_data)
    probablity = best_model.predict_proba(processed_data)[:,1]

    if single == 1:
        return "This customer is likely to be churned with a confidence of {:.2f}%".format(probablity*100)
    else:
        return "This customer is likely to continue with a confidence of {:.2f}%".format(probablity*100)

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(label='SeniorCitizen', choices=[0, 1]),
        gr.Number(label='MonthlyCharges'),
        gr.Number(label='TotalCharges'),
        gr.Dropdown(label='gender', choices=["Male", "Female"]),
        gr.Dropdown(label='Partner', choices=["Yes", "No"]),
        gr.Dropdown(label='Dependents', choices=["Yes", "No"]),
        gr.Dropdown(label='PhoneService', choices=["Yes", "No"]),
        gr.Dropdown(label='MultipleLines', choices=["Yes", "No", "No phone service"]),
        gr.Dropdown(label='InternetService', choices=["DSL", "Fiber optic", "No"]),
        gr.Dropdown(label='OnlineSecurity', choices=["Yes", "No", "No internet service"]),
        gr.Dropdown(label='OnlineBackup', choices=["Yes", "No", "No internet service"]),
        gr.Dropdown(label='DeviceProtection', choices=["Yes", "No", "No internet service"]),
        gr.Dropdown(label='TechSupport', choices=["Yes", "No", "No internet service"]),
        gr.Dropdown(label='StreamingTV', choices=["Yes", "No", "No internet service"]),
        gr.Dropdown(label='StreamingMovies', choices=["Yes", "No", "No internet service"]),
        gr.Dropdown(label='Contract', choices=["Month-to-month", "One year", "Two year"]),
        gr.Dropdown(label='PaperlessBilling', choices=["Yes", "No"]),
        gr.Dropdown(label='PaymentMethod', choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
        gr.Number(label='tenure: 0-72')
    ],
    outputs="text",
    theme="dark",
    title="Customer Churn Prediction",
    description="Predict whether a customer is likely to churn."
)

iface.launch(share=True)
