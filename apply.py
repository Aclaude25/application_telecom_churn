import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Chargement du modèle et du scaler
with open('churn_model1.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

# Chargement des colonnes d'entrée utilisées lors de l'entraînement
features = ['tenure', 'InternetService_Fiber optic', 'InternetService_No',
            'OnlineSecurity_No internet service', 'OnlineBackup_No internet service',
            'DeviceProtection_No internet service', 'TechSupport_No internet service',
            'StreamingTV_No internet service', 'StreamingMovies_No internet service',
            'Contract_Two year', 'PaymentMethod_Electronic check']

# Page principale
st.set_page_config(page_title="Prédiction du Churn", layout="centered")
st.title("📉 Prédiction du Churn Client Télécom")
st.markdown("""
Bienvenue dans l'application de prédiction du **churn client** dans le secteur des télécommunications.

Ce projet vise à aider les entreprises à **anticiper la perte de clients** en analysant certains facteurs liés à leurs services.
Les données proviennent d’un jeu de données public de Telco. Le modèle utilisé est une **régression logistique**,
entraînée sur des variables les plus corrélées avec le churn.
""")

st.sidebar.header("🛠️ Données client à renseigner")

# Entrées utilisateur
tenure = st.sidebar.slider("Durée d'abonnement (mois)", 0, 72, 12)
internet_fiber = st.sidebar.checkbox("Internet par fibre optique")
internet_no = st.sidebar.checkbox("Pas de service internet")
online_security_no = st.sidebar.checkbox("Pas de sécurité en ligne")
online_backup_no = st.sidebar.checkbox("Pas de sauvegarde en ligne")
device_protection_no = st.sidebar.checkbox("Pas de protection d'appareil")
tech_support_no = st.sidebar.checkbox("Pas de support technique")
streaming_tv_no = st.sidebar.checkbox("Pas de TV en streaming")
streaming_movies_no = st.sidebar.checkbox("Pas de films en streaming")
contract_two_year = st.sidebar.checkbox("Contrat de deux ans")
payment_electronic_check = st.sidebar.checkbox("Paiement par chèque électronique")

# Création des données d'entrée
input_data = pd.DataFrame([{
    'tenure': tenure,
    'InternetService_Fiber optic': int(internet_fiber),
    'InternetService_No': int(internet_no),
    'OnlineSecurity_No internet service': int(online_security_no),
    'OnlineBackup_No internet service': int(online_backup_no),
    'DeviceProtection_No internet service': int(device_protection_no),
    'TechSupport_No internet service': int(tech_support_no),
    'StreamingTV_No internet service': int(streaming_tv_no),
    'StreamingMovies_No internet service': int(streaming_movies_no),
    'Contract_Two year': int(contract_two_year),
    'PaymentMethod_Electronic check': int(payment_electronic_check)
}])

# Imputation (si nécessaire)
imputer = SimpleImputer(strategy='median')
input_data_imputed = pd.DataFrame(imputer.fit_transform(input_data), columns=features)

# Standardisation
input_data_scaled = scaler.transform(input_data_imputed)

# Prédiction
if st.button("🔍 Prédire le risque de churn"):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0, 1]

    if prediction[0] == 1:
        st.error(f"⚠️ Le client est à risque de churn avec une probabilité de {probability:.2%}")
    else:
        st.success(f"✅ Le client n'est pas à risque de churn avec une probabilité de {(1 - probability):.2%}")

# Informations supplémentaires
st.markdown("---")
with st.expander("ℹ️ À propos du modèle"):
    st.markdown("""
    - **Modèle utilisé** : Régression Logistique
    - **Données d'entraînement** : jeu de données Telco Customer Churn
    - **variables à forte correlation avec la variable cybe Churn** : **tenure** = Nombre de mois que le client a passé avec l'entreprise -     
                                                                      **InternetService_Fiber optic** = Fourniture de services Internet (Fibre optique) du client - 
                                                                      **InternetService_No** = Pas Fourniture de services Internet du client -  
                                                                      **OnlineSecurity_No internet service** = le client n'a pas une sécurité en ligne - 
                                                                      **OnlineBackup_No internet service** = le client n'a pas une sauvegarde en ligne -   
                                                                      **DeviceProtection_No internet service** = le client n'a pas une protection des appareils -  
                                                                      **TechSupport_No internet service** = le client n'a pas un support technique -  
                                                                      **StreamingTV_No internet service** = le client n'a pas la télévision en streaming -  
                                                                      **StreamingMovies_No internet service** = le client n'a pas des films en streaming -  
                                                                      **Contract_Two year** = la durée du contrat du client -  
                                                                      **PaymentMethod_Electronic check** = le mode de paiement du client
    - **Prétraitements** : imputation des valeurs manquantes, normalisation, encodage catégoriel
    """)

with st.expander("📊 Objectif de l'application"):
    st.markdown("""
    - L’objectif est de fournir un outil de prédiction simple et interactif permettant aux responsables de fidélisation client d’identifier les profils à risque, et de prendre des décisions proactives.
    
    - **ci suit les variables correspondent aux boutons**:  **tenure** = tenure - 
                                                            **InternetService_Fiber optic** = internet_fiber - 
                                                            **InternetService_No** = internet_no - 
                                                            **OnlineSecurity_No internet service** = online_security_no - 
                                                            **OnlineBackup_No internet service** = online_backup_no -
                                                            **DeviceProtection_No internet service** = device_protection_no - 
                                                            **TechSupport_No internet service** = tech_support_no - 
                                                            **StreamingTV_No internet service** = streaming_tv_no - 
                                                            **StreamingMovies_No internet service** = streaming_movies_no - 
                                                            **Contract_Two year** = contract_two_year - 
                                                            **PaymentMethod_Electronic check** = payment_electronic_check - 
    """)
