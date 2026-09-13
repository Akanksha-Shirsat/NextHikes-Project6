import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load Model and Store Data
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/final_random_forest_pipeline.pkl")
    return model


@st.cache_data
def load_store_data():
    store = pd.read_csv("data/store.csv")
    return store


rf_pipeline = load_model()
store_df = load_store_data()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Rossmann Store Sales Forecasting")

st.write(
    "Predict daily sales for a Rossmann store using "
    "store characteristics, promotions, holidays, competition, "
    "and date-based features."
)


# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------

st.sidebar.header("Prediction Inputs")

store_id = st.sidebar.selectbox(
    "Select Store",
    sorted(store_df["Store"].unique())
)

prediction_date = st.sidebar.date_input(
    "Select Date"
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[1, 2, 3, 4, 5, 6, 7],
    index=0,
    help="1 = Monday, 7 = Sunday"
)

open_status = st.sidebar.selectbox(
    "Store Open",
    options=[0, 1],
    index=1
)

promo = st.sidebar.selectbox(
    "Promotion",
    options=[0, 1],
    index=0
)

state_holiday = st.sidebar.selectbox(
    "State Holiday",
    options=["None", "a", "b", "c"]
)

school_holiday = st.sidebar.selectbox(
    "School Holiday",
    options=[0, 1],
    index=0
)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.sidebar.button("Predict Sales"):

    selected_store = store_df[
        store_df["Store"] == store_id
    ].copy()

    # Date features
    prediction_date = pd.Timestamp(prediction_date)

    year = prediction_date.year
    month = prediction_date.month
    day = prediction_date.day
    week_of_year = prediction_date.isocalendar().week
    is_weekend = int(day_of_week >= 6)

    # Store information
    store_info = selected_store.iloc[0]

    # Competition information
    competition_distance = store_info["CompetitionDistance"]

    if pd.isna(competition_distance):
        competition_distance = store_df["CompetitionDistance"].median()

    competition_month = store_info["CompetitionOpenSinceMonth"]

    if pd.isna(competition_month):
        competition_month = 0

    competition_year = store_info["CompetitionOpenSinceYear"]

    if pd.isna(competition_year):
        competition_year = 0

    competition_open_months = 0

    if competition_year > 0 and competition_month > 0:

        competition_open_months = (
            (year - competition_year) * 12
            + (month - competition_month)
        )

        competition_open_months = max(
            competition_open_months,
            0
        )

    # Promo2 information
    promo2_since_week = store_info["Promo2SinceWeek"]

    if pd.isna(promo2_since_week):
        promo2_since_week = 0

    promo2_since_year = store_info["Promo2SinceYear"]

    if pd.isna(promo2_since_year):
        promo2_since_year = 0

    promo_interval = store_info["PromoInterval"]

    if pd.isna(promo_interval):
        promo_interval = "None"

    promo2_active = int(store_info["Promo2"])

    # State holiday standardization
    if state_holiday == "None":
        state_holiday = "None"

    # --------------------------------------------------
    # Create Model Input
    # --------------------------------------------------

    input_data = pd.DataFrame({
        "Store": [store_id],
        "DayOfWeek": [day_of_week],
        "Date": [prediction_date],
        "Open": [open_status],
        "Promo": [promo],
        "StateHoliday": [state_holiday],
        "SchoolHoliday": [school_holiday],
        "StoreType": [store_info["StoreType"]],
        "Assortment": [store_info["Assortment"]],
        "CompetitionDistance": [competition_distance],
        "CompetitionOpenSinceMonth": [competition_month],
        "CompetitionOpenSinceYear": [competition_year],
        "Promo2": [store_info["Promo2"]],
        "Promo2SinceWeek": [promo2_since_week],
        "Promo2SinceYear": [promo2_since_year],
        "PromoInterval": [promo_interval],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "WeekOfYear": [int(week_of_year)],
        "IsWeekend": [is_weekend],
        "CompetitionOpenMonths": [competition_open_months],
        "Promo2Active": [promo2_active]
    })


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = rf_pipeline.predict(input_data)

    predicted_sales = max(
        float(prediction[0]),
        0
    )


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success("Sales prediction generated successfully!")

    st.metric(
        label="Predicted Daily Sales",
        value=f"₹{predicted_sales:,.2f}"
    )


    # --------------------------------------------------
    # Store Information
    # --------------------------------------------------

    st.subheader("Store Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Store",
        store_id
    )

    col2.metric(
        "Store Type",
        store_info["StoreType"]
    )

    col3.metric(
        "Assortment",
        store_info["Assortment"]
    )

    col4.metric(
        "Competition Distance",
        f"{competition_distance:.2f}"
    )


    # --------------------------------------------------
    # Prediction Details
    # --------------------------------------------------

    st.subheader("Prediction Details")

    details = pd.DataFrame({
        "Feature": [
            "Date",
            "Day of Week",
            "Store Open",
            "Promotion",
            "State Holiday",
            "School Holiday",
            "Promo2"
        ],
        "Value": [
            prediction_date.strftime("%Y-%m-%d"),
            day_of_week,
            open_status,
            promo,
            state_holiday,
            school_holiday,
            promo2_active
        ]
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Rossmann Store Sales Forecasting | "
    "Random Forest Regression Model"
)