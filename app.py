import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit title
st.title("Titanic Survival Prediction")

# File uploader
file = st.file_uploader("Upload Titanic CSV", type="csv")

if file:
    # Read dataset
    df = pd.read_csv(file)

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df.dropna(subset=['Embarked'], inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # Features and target
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male',
                'Embarked_Q', 'Embarked_S']
    X = df[features]
    y = df['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
