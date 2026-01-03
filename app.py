import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import seaborn as sns

# ---- Try importing TensorFlow/Keras for ANN ----
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(page_title="ML Model Explorer", layout="wide")

st.title("Machine Learning Model Explorer")
st.write("""
Upload your dataset, choose **Regression** or **Classification**, 
select a model, and the app will:
- Train & test the model  
- Show performance metrics  
- Display coefficients / feature importances where applicable  
- Allow prediction for new input values
""")

# ------------------ Sidebar: Problem Setup ------------------
st.sidebar.header("1️⃣ Data & Problem Setup")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["Regression", "Classification"]
)

test_size = st.sidebar.slider(
    "Test Size (fraction for testing)",
    min_value=0.1, max_value=0.4, value=0.2, step=0.05
)

# Model selection based on problem type
if problem_type == "Regression":
    model_name = st.sidebar.selectbox(
        "Select Regression Model",
        [
            "Linear Regression",
            "Polynomial Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "Gradient Boosting Regressor",
            "ANN (TensorFlow/Keras)"
        ]
    )
else:
    model_name = st.sidebar.selectbox(
        "Select Classification Model",
        [
            "Logistic Regression",
            "Decision Tree Classifier",
            "Random Forest Classifier",
            "Gradient Boosting Classifier"
        ]
    )

# Extra parameter for Polynomial Regression
poly_degree = None
if problem_type == "Regression" and model_name == "Polynomial Regression":
    poly_degree = st.sidebar.slider(
        "Polynomial Degree",
        min_value=2, max_value=6, value=2, step=1
    )

# Extra basic ANN settings
if problem_type == "Regression" and model_name == "ANN (TensorFlow/Keras)":
    if TF_AVAILABLE:
        epochs = st.sidebar.slider("ANN Epochs", 10, 300, 100, 10)
        neurons = st.sidebar.slider("Neurons per Hidden Layer", 8, 128, 32, 8)
    else:
        st.sidebar.warning("TensorFlow not installed. ANN option will not run.")


# ------------------ Main Content ------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.write(df.head())

    st.markdown("---")

    # Column selection
    all_columns = list(df.columns)

    target_col = st.selectbox(
        "Select Target (Output) Column",
        all_columns
    )

    feature_cols = st.multiselect(
        "Select Input Feature Columns (X)",
        [c for c in all_columns if c != target_col],
        default=[c for c in all_columns if c != target_col]
    )

    if len(feature_cols) == 0:
        st.error("Please select at least one feature column.")
    else:
        X = df[feature_cols]
        y = df[target_col]

        # Basic type display
        st.write("**Selected Features (X):**", feature_cols)
        st.write("**Target (y):**", target_col)
        st.write("Number of samples:", len(df))

        # ------------------ Suggest a Model ------------------
        st.markdown("### 🤖 Model Suggestion (Basic Heuristic)")
        n_samples, n_features = X.shape

        if problem_type == "Regression":
            if n_samples < 50:
                st.info("Dataset is small → try **Linear Regression** or **Decision Tree Regressor**.")
            elif n_samples < 500:
                st.info("Medium-sized dataset → **Random Forest Regressor** or **Gradient Boosting Regressor** often perform well.")
            else:
                st.info("Large dataset → **Gradient Boosting Regressor** / **ANN** can be good choices.")
        else:
            n_classes = df[target_col].nunique()
            if n_classes == 2:
                st.info("Binary classification detected → **Logistic Regression**, **Random Forest**, or **Gradient Boosting** are good choices.")
            else:
                st.info(f"Multi-class ({n_classes} classes) → **Random Forest Classifier** or **Gradient Boosting Classifier** recommended.")

        # ------------------ Train-Test Split ------------------
        # Handle non-numeric features: simple encoding (drop non-numeric or use get_dummies)
        # For simplicity, use get_dummies on X
        X_encoded = pd.get_dummies(X, drop_first=True)

        # For classification, encode target if non-numeric
        from sklearn.preprocessing import LabelEncoder
        y_encoded = y.copy()
        label_encoder = None
        if problem_type == "Classification":
            if y.dtype == "object" or str(y.dtype).startswith("category"):
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            else:
                y_encoded = y.values
        else:
            y_encoded = y.values

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size, random_state=42
        )

        st.markdown("### 📊 Train/Test Split")
        st.write(f"Train samples: {X_train.shape[0]}")
        st.write(f"Test samples: {X_test.shape[0]}")

        # ANN needs scaling, others can use raw X
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ------------------ Train Model ------------------
        train_button = st.button("🚀 Train Model")

        if train_button:
            model = None
            history = None  # for ANN

            # ===== Regression Models =====
            if problem_type == "Regression":

                if model_name == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                elif model_name == "Polynomial Regression":
                    poly = PolynomialFeatures(degree=poly_degree)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)

                    model = LinearRegression()
                    model.fit(X_train_poly, y_train)

                    # Override X_train/X_test for metric computation
                    X_train_used = X_train_poly
                    X_test_used = X_test_poly
                elif model_name == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(random_state=42)
                    model.fit(X_train, y_train)

                elif model_name == "Random Forest Regressor":
                    model = RandomForestRegressor(random_state=42, n_estimators=200)
                    model.fit(X_train, y_train)

                elif model_name == "Gradient Boosting Regressor":
                    model = GradientBoostingRegressor(random_state=42)
                    model.fit(X_train, y_train)

                elif model_name == "ANN (TensorFlow/Keras)":
                    if not TF_AVAILABLE:
                        st.error("TensorFlow/Keras is not installed in this environment.")
                    else:
                        input_dim = X_train_scaled.shape[1]
                        model = Sequential()
                        model.add(Dense(neurons, activation='relu', input_dim=input_dim))
                        model.add(Dense(neurons, activation='relu'))
                        model.add(Dense(1))  # regression output

                        model.compile(optimizer='adam', loss='mse')
                        history = model.fit(
                            X_train_scaled, y_train,
                            epochs=epochs, batch_size=32,
                            validation_split=0.2,
                            verbose=0
                        )

                # ----- Predictions & Metrics (Regression) -----
                if model_name == "Polynomial Regression":
                    y_train_pred = model.predict(X_train_used)
                    y_test_pred = model.predict(X_test_used)
                elif model_name == "ANN (TensorFlow/Keras)" and TF_AVAILABLE:
                    y_train_pred = model.predict(X_train_scaled).flatten()
                    y_test_pred = model.predict(X_test_scaled).flatten()
                else:
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                st.markdown("### 📈 Regression Metrics")
                st.write(f"**Train R²**: {r2_score(y_train, y_train_pred):.4f}")
                st.write(f"**Test R²**: {r2_score(y_test, y_test_pred):.4f}")
                st.write(f"**Test RMSE**: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
                st.write(f"**Test MAE**: {mean_absolute_error(y_test, y_test_pred):.4f}")

                # Plot predicted vs actual
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_test_pred, alpha=0.7)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted (Test Set)")
                st.pyplot(fig)

                # Coefficients or feature importances
                st.markdown("### 🔍 Model Details")
                if model_name in ["Linear Regression", "Polynomial Regression"] and hasattr(model, "coef_"):
                    st.write("Coefficients (on encoded feature space):")
                    if model_name == "Polynomial Regression":
                        # Polynomial feature names
                        poly_feature_names = poly.get_feature_names_out(X_train.columns)
                        coef_df = pd.DataFrame({
                            "Feature": poly_feature_names,
                            "Coefficient": model.coef_
                        })
                    else:
                        coef_df = pd.DataFrame({
                            "Feature": X_train.columns,
                            "Coefficient": model.coef_
                        })
                    st.dataframe(coef_df)
                elif "Tree" in model_name or "Forest" in model_name or "Boosting" in model_name:
                    if hasattr(model, "feature_importances_"):
                        fi_df = pd.DataFrame({
                            "Feature": X_train.columns,
                            "Importance": model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)
                        st.write("Feature Importances:")
                        st.dataframe(fi_df)

                # Optional: ANN loss curve
                if model_name == "ANN (TensorFlow/Keras)" and TF_AVAILABLE and history is not None:
                    st.markdown("### 🧠 ANN Training Loss Curve")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(history.history['loss'], label='Train Loss')
                    ax2.plot(history.history['val_loss'], label='Val Loss')
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("MSE Loss")
                    ax2.set_title("ANN Training History")
                    ax2.legend()
                    st.pyplot(fig2)

            # ===== Classification Models =====
            else:
                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)

                elif model_name == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(random_state=42)
                    model.fit(X_train, y_train)

                elif model_name == "Random Forest Classifier":
                    model = RandomForestClassifier(random_state=42, n_estimators=200)
                    model.fit(X_train, y_train)

                elif model_name == "Gradient Boosting Classifier":
                    model = GradientBoostingClassifier(random_state=42)
                    model.fit(X_train, y_train)

                # Predictions & Metrics (Classification)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                st.markdown("### 📊 Classification Metrics")
                st.write(f"**Train Accuracy**: {accuracy_score(y_train, y_train_pred):.4f}")
                st.write(f"**Test Accuracy**: {accuracy_score(y_test, y_test_pred):.4f}")
                st.write(f"**Test Precision (macro)**: {precision_score(y_test, y_test_pred, average='macro', zero_division=0):.4f}")
                st.write(f"**Test Recall (macro)**: {recall_score(y_test, y_test_pred, average='macro', zero_division=0):.4f}")
                st.write(f"**Test F1-score (macro)**: {f1_score(y_test, y_test_pred, average='macro', zero_division=0):.4f}")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                ax_cm.set_title("Confusion Matrix")
                st.pyplot(fig_cm)

                st.markdown("### 🧾 Classification Report")
                st.text(classification_report(y_test, y_test_pred, zero_division=0))

                # Coefficients / Feature Importances
                st.markdown("### 🔍 Model Details")
                if model_name == "Logistic Regression" and hasattr(model, "coef_"):
                    coef_df = pd.DataFrame(model.coef_, columns=X_train.columns).T
                    coef_df.columns = [f"Class {i}" for i in range(coef_df.shape[1])]
                    coef_df["Feature"] = X_train.columns
                    st.dataframe(coef_df.set_index("Feature"))
                elif ("Tree" in model_name or "Forest" in model_name or "Boosting" in model_name) and hasattr(model, "feature_importances_"):
                    fi_df = pd.DataFrame({
                        "Feature": X_train.columns,
                        "Importance": model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)
                    st.write("Feature Importances:")
                    st.dataframe(fi_df)

            # ------------------ New Prediction ------------------
            st.markdown("## 🔮 Predict on New Input")

            with st.form("prediction_form"):
                st.write("Enter values for each feature:")
                new_data = {}
                for col in X.columns:
                    # try numeric input; if non-numeric in original, ask as text
                    if pd.api.types.is_numeric_dtype(X[col]):
                        val = st.number_input(f"{col}", value=float(X[col].mean()))
                    else:
                        val = st.text_input(f"{col}", value=str(X[col].iloc[0]))
                    new_data[col] = val

                predict_btn = st.form_submit_button("Predict")

            if predict_btn:
                new_df = pd.DataFrame([new_data])

                # Match training encoding
                new_df_encoded = pd.get_dummies(new_df)
                # Add missing columns (if any)
                for col in X_encoded.columns:
                    if col not in new_df_encoded.columns:
                        new_df_encoded[col] = 0
                new_df_encoded = new_df_encoded[X_encoded.columns]  # same column order

                if problem_type == "Regression":
                    if model_name == "Polynomial Regression":
                        new_df_poly = poly.transform(new_df_encoded)
                        y_new = model.predict(new_df_poly)
                    elif model_name == "ANN (TensorFlow/Keras)" and TF_AVAILABLE:
                        new_scaled = scaler.transform(new_df_encoded)
                        y_new = model.predict(new_scaled).flatten()
                    else:
                        y_new = model.predict(new_df_encoded)
                    st.success(f"Predicted output (y) = {y_new[0]:.4f}")
                else:
                    y_new = model.predict(new_df_encoded)
                    if label_encoder is not None:
                        y_new_label = label_encoder.inverse_transform(y_new)[0]
                        st.success(f"Predicted class = {y_new_label}")
                    else:
                        st.success(f"Predicted class = {y_new[0]}")

else:
    st.info("Please upload a CSV file to begin.")
