"""
House Price Prediction – Streamlit + scikit-learn mini-project
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_LABELS = {
    "MedInc": "Median Income (in $10,000s)",
    "HouseAge": "House Age (years)",
    "AveRooms": "Average Rooms per Household",
    "AveBedrms": "Average Bedrooms per Household",
    "Population": "Block Population",
    "AveOccup": "Average Occupants per Household",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}


@st.cache_data(show_spinner="Loading California Housing dataset...")
def load_data():
    housing = fetch_california_housing(as_frame=True)
    return housing.data, housing.target


@st.cache_resource(show_spinner="Training models (first run only)...")
def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "R² Score": r2_score(y_test, preds),
                "MAE ($)": mean_absolute_error(y_test, preds) * 100_000,
            }
        )

    metrics = pd.DataFrame(rows).set_index("Model")
    return models, metrics


def build_sidebar(X, model_names):
    st.sidebar.header("⚙️ Model")
    selected_model = st.sidebar.selectbox("Regression algorithm", model_names)

    st.sidebar.header("🏡 House Features")
    values = {}
    for col, label in FEATURE_LABELS.items():
        low = float(X[col].quantile(0.01))
        high = float(X[col].quantile(0.99))
        default = float(X[col].median())
        step = 1.0 if col == "Population" else 0.01
        values[col] = st.sidebar.slider(
            label, min_value=low, max_value=high, value=default, step=step
        )

    return selected_model, pd.DataFrame([values], columns=X.columns)


def main():
    st.title("🏠 House Price Prediction")
    st.caption(
        "Machine-learning regression on the California Housing dataset. "
        "Pick an algorithm and tweak the house features in the sidebar."
    )

    X, _ = load_data()
    models, metrics = train_models()
    selected_model, user_input = build_sidebar(X, list(models.keys()))

    prediction = models[selected_model].predict(user_input)[0] * 100_000
    prediction = max(prediction, 0)

    st.subheader("💰 Predicted House Price")
    st.metric(label=f"Estimated median value ({selected_model})", value=f"${prediction:,.0f}")

    st.divider()

    st.subheader(f"📊 Performance: {selected_model} (test split)")
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{metrics.loc[selected_model, 'R² Score']:.3f}")
    col2.metric("Mean Absolute Error", f"${metrics.loc[selected_model, 'MAE ($)']:,.0f}")

    st.subheader("🔍 Model Comparison")
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(
            metrics.style.format({"R² Score": "{:.3f}", "MAE ($)": "${:,.0f}"}),
            use_container_width=True,
        )
    with right:
        st.bar_chart(metrics["R² Score"])

    with st.expander("See your input values and all model predictions"):
        st.dataframe(user_input, use_container_width=True, hide_index=True)
        all_preds = {
            name: f"${max(m.predict(user_input)[0] * 100_000, 0):,.0f}"
            for name, m in models.items()
        }
        st.table(pd.Series(all_preds, name="Predicted Price"))

    st.caption(
        f"Trained on {int(len(X) * (1 - TEST_SIZE)):,} samples, "
        f"evaluated on {int(len(X) * TEST_SIZE):,} held-out samples."
    )


if __name__ == "__main__":
    main()