import gradio as gr
import pandas as pd
import pickle

def load_ml_components(fp):
    "Load the ml component to reuse in App"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object

model = load_ml_components("model.sav")
pipeline = load_ml_components("pipeline.sav")

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
                                         
    # Process the 'tenure' column
    if new_df['tenure'].isnull().any():
        new_df['tenure'].fillna(new_df['tenure'].median(), inplace=True)
    new_df['tenure'] = new_df['tenure'].astype(int)

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_df['tenure_group'] = pd.cut(new_df['tenure'], range(1, 80, 12), right=False, labels=labels)
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    # Transforming the data using the pipeline
    processed_data = pipeline.transform(new_df)
    
    single = model.predict(processed_data)
    probablity = model.predict_proba(processed_data)[:,1]
    
    if single == 1:
        return "This customer is likely to be churned with a confidence of {:.2f}%".format(probablity*100)
    else:
        return "This customer is likely to continue with a confidence of {:.2f}%".format(probablity*100)

# Define the Gradio interface
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
        gr.Number(label='tenure: 0-72', min_value=0, max_value=72)
    ],
    outputs="text",
    live=True
)

iface.launch(share=True)
