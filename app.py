import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Function to preprocess data
def preprocess_data(df):
    if df is None or df.empty:
        st.error("Error: The DataFrame is None or empty after loading.")
        return None

    # Convert 'Login Time' to datetime and extract 'Login Hour'
    if "Login Time" in df.columns:
        df["Login Time"] = pd.to_datetime(df["Login Time"], errors="coerce")
        df["Login Hour"] = df["Login Time"].dt.hour
    else:
        st.error("Error: 'Login Time' column is missing in the dataset.")
        return None

    # Calculate 'Survey Participation Rate'
    if "Count of Survey Attempts" in df.columns:
        df["Survey Participation Rate"] = df["Count of Survey Attempts"] / (df["Count of Survey Attempts"] + 1)
    else:
        st.error("Error: 'Count of Survey Attempts' column is missing.")
        return None

    # Ensure 'NPI' exists before grouping
    if "NPI" in df.columns and "Login Hour" in df.columns:
        df["Active Hours"] = df.groupby("NPI")["Login Hour"].transform(lambda x: x.mode()[0] if not x.mode().empty else x.mean())
    else:
        st.error("Error: 'NPI' or 'Login Hour' column is missing.")
        return None

    # Drop unnecessary columns safely
    drop_cols = ["Login Time", "Logout Time"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # One-hot encode categorical features
    categorical_columns = ["State", "Region", "Speciality"]
    df = pd.get_dummies(df, columns=[col for col in categorical_columns if col in df.columns], drop_first=True)

    return df

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dummy_npi_data.csv")  # Ensure correct file extension
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Main function for Streamlit app
def main():
    st.title("Doctor Survey Prediction App")

    df = load_data()

    if df is None:
        st.error("Dataset could not be loaded.")
        return

    df = preprocess_data(df)

    if df is None or df.empty:
        st.error("Preprocessing failed. Check dataset structure.")
        return

    # Split data
    target_col = "Survey Participation Rate"
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        st.error(f"Error: '{target_col}' column is missing in the dataset.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a regression model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Streamlit UI
    st.subheader("Make a Prediction")
    login_hour = st.slider("Select Login Hour", 0, 23, 12)

    # Prepare input for prediction
    sample_input = X_test.iloc[0:1].copy()
    sample_input["Login Hour"] = login_hour

    prediction = model.predict(sample_input)
    st.write(f"Predicted Survey Participation Rate: {prediction[0]:.2f}")

    # Download index positions of predicted doctors
    st.subheader("Download Index Positions of Predicted Doctors")
    
    filtered_df = df[df["Login Hour"] == login_hour]  # Example filtering
    
    if not filtered_df.empty:
        index_positions = filtered_df.index.to_frame(index=False)  # Extract only index positions
        csv = index_positions.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Index Positions as CSV",
            data=csv,
            file_name="index_positions.csv",
            mime="text/csv"
        )
    else:
        st.warning("No doctors found for the selected login hour.")

if __name__ == "__main__":
    main()
