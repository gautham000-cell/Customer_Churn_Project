
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Used Vehicle Price Intelligence",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "final_vehicle_price_model.pkl"
PREPROCESSOR_PATH = "vehicle_preprocessor.pkl"
MARKET_DATA_PATH = "market_intelligence_data.csv"
PERFORMANCE_PATH = "model_performance.json"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

@st.cache_resource
def load_preprocessor():

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return preprocessor


# ============================================================
# LOAD MARKET DATA
# ============================================================

@st.cache_data
def load_market_data():

    data = pd.read_csv(MARKET_DATA_PATH)

    return data


# ============================================================
# LOAD MODEL PERFORMANCE
# ============================================================

@st.cache_data
def load_performance():

    try:

        with open(PERFORMANCE_PATH, "r") as file:
            performance = json.load(file)

        return performance

    except Exception:

        return {}


# ============================================================
# LOAD ALL FILES
# ============================================================

try:

    model = load_model()
    preprocessor = load_preprocessor()
    market_data = load_market_data()
    performance = load_performance()

except Exception as error:

    st.error("Unable to load the project files.")
    st.error(str(error))
    st.stop()


# ============================================================
# HELPER FUNCTION - MARKET BENCHMARK
# ============================================================

def find_market_benchmark(
    brand,
    fuel,
    transmission,
    vehicle_age
):

    data = market_data.copy()

    # --------------------------------------------------------
    # Check expected columns
    # --------------------------------------------------------

    possible_price_columns = [
        "Market_Average_Price",
        "market_average_price",
        "selling_price",
        "Selling_Price"
    ]

    price_column = None

    for column in possible_price_columns:

        if column in data.columns:
            price_column = column
            break

    if price_column is None:

        return None, 0, None

    # --------------------------------------------------------
    # Exact comparable group
    # --------------------------------------------------------

    comparable = data[
        (data["brand"].astype(str).str.lower() == str(brand).lower()) &
        (data["fuel"].astype(str).str.lower() == str(fuel).lower()) &
        (
            data["transmission"].astype(str).str.lower()
            == str(transmission).lower()
        )
    ].copy()

    if len(comparable) == 0:

        return None, 0, None

    # --------------------------------------------------------
    # Find closest vehicle age
    # --------------------------------------------------------

    if "vehicle_age" in comparable.columns:

        comparable["age_difference"] = (
            comparable["vehicle_age"] - vehicle_age
        ).abs()

        closest = comparable.sort_values(
            "age_difference"
        ).head(1)

        benchmark_price = float(
            closest[price_column].iloc[0]
        )

        if "vehicle_count" in closest.columns:

            vehicle_count = int(
                closest["vehicle_count"].iloc[0]
            )

        elif "count" in closest.columns:

            vehicle_count = int(
                closest["count"].iloc[0]
            )

        else:

            vehicle_count = len(comparable)

        benchmark_age = int(
            closest["vehicle_age"].iloc[0]
        )

        return benchmark_price, vehicle_count, benchmark_age

    return None, 0, None


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🚗 Used Vehicle Price Prediction & Market Intelligence")

st.write(
    """
    An end-to-end Data Science application that predicts the selling
    price of a used vehicle and compares it with comparable vehicles
    in the market.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚗 Vehicle Information")

st.sidebar.write(
    "Enter the vehicle details below to estimate its market value."
)


# ============================================================
# VEHICLE INPUTS
# ============================================================

brand = st.sidebar.text_input(
    "Brand",
    value="Maruti"
)

model_name = st.sidebar.text_input(
    "Model",
    value="Swift"
)

year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2018,
    step=1
)

km_driven = st.sidebar.number_input(
    "Kilometres Driven",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000
)

fuel = st.sidebar.selectbox(
    "Fuel Type",
    [
        "Diesel",
        "Petrol",
        "CNG",
        "LPG"
    ]
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    [
        "Individual",
        "Dealer",
        "Trustmark Dealer"
    ]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic"
    ]
)

owner = st.sidebar.selectbox(
    "Owner Type",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above",
        "Test Drive Car"
    ]
)

mileage = st.sidebar.number_input(
    "Mileage (km/l)",
    min_value=1.0,
    max_value=100.0,
    value=18.0,
    step=0.1
)

engine_cc = st.sidebar.number_input(
    "Engine (CC)",
    min_value=500.0,
    max_value=10000.0,
    value=1200.0,
    step=50.0
)

max_power_bhp = st.sidebar.number_input(
    "Maximum Power (BHP)",
    min_value=20.0,
    max_value=2000.0,
    value=80.0,
    step=1.0
)

torque_nm = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=20.0,
    max_value=2000.0,
    value=110.0,
    step=1.0
)

seats = st.sidebar.number_input(
    "Number of Seats",
    min_value=2,
    max_value=15,
    value=5,
    step=1
)

asking_price = st.sidebar.number_input(
    "Seller Asking Price (₹)",
    min_value=0,
    max_value=100000000,
    value=500000,
    step=10000
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "🔍 Predict Vehicle Price",
    use_container_width=True
)


# ============================================================
# APPLICATION TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "💰 Price Prediction",
        "📊 Market Intelligence",
        "ℹ️ About the Model"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Calculate vehicle age
    # --------------------------------------------------------

    vehicle_age = 2026 - int(year)

    if vehicle_age < 0:

        st.error("Manufacturing year cannot be in the future.")
        st.stop()

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    safe_age = vehicle_age if vehicle_age > 0 else 1

    km_per_year = km_driven / safe_age

    power_per_engine_cc = (
        max_power_bhp / engine_cc
        if engine_cc > 0
        else 0
    )

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "year": [year],

        "vehicle_age": [vehicle_age],

        "km_driven": [km_driven],

        "mileage": [mileage],

        "engine_cc": [engine_cc],

        "max_power_bhp": [max_power_bhp],

        "torque_nm": [torque_nm],

        "seats": [seats],

        "km_per_year": [km_per_year],

        "power_per_engine_cc": [power_per_engine_cc],

        "brand": [brand],

        "model": [model_name],

        "fuel": [fuel],

        "seller_type": [seller_type],

        "transmission": [transmission],

        "owner": [owner]
    })

    # --------------------------------------------------------
    # Transform input
    # --------------------------------------------------------

    try:

        processed_input = preprocessor.transform(
            input_data
        )

        predicted_price = model.predict(
            processed_input
        )[0]

        predicted_price = max(
            0,
            float(predicted_price)
        )

    except Exception as error:

        st.error(
            "Prediction failed. Please check the vehicle details."
        )

        st.error(str(error))

        st.stop()

    # ========================================================
    # MARKET BENCHMARK
    # ========================================================

    benchmark_price, comparable_count, benchmark_age = (
        find_market_benchmark(
            brand,
            fuel,
            transmission,
            vehicle_age
        )
    )

    # ========================================================
    # TAB 1 - PRICE PREDICTION
    # ========================================================

    with tab1:

        st.subheader("💰 Estimated Vehicle Price")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "ML Predicted Price",
                f"₹{predicted_price:,.0f}"
            )

        with col2:

            if benchmark_price is not None:

                st.metric(
                    "Market Benchmark",
                    f"₹{benchmark_price:,.0f}"
                )

            else:

                st.metric(
                    "Market Benchmark",
                    "Not Available"
                )

        with col3:

            if benchmark_price is not None:

                difference = (
                    predicted_price - benchmark_price
                )

                st.metric(
                    "Prediction Difference",
                    f"₹{difference:,.0f}"
                )

            else:

                st.metric(
                    "Prediction Difference",
                    "N/A"
                )

        st.divider()

        # ----------------------------------------------------
        # Vehicle summary
        # ----------------------------------------------------

        st.subheader("🚘 Vehicle Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(f"**Brand:** {brand}")
            st.write(f"**Model:** {model_name}")
            st.write(f"**Year:** {year}")
            st.write(f"**Vehicle Age:** {vehicle_age} years")
            st.write(f"**Kilometres Driven:** {km_driven:,} km")

        with summary_col2:

            st.write(f"**Fuel:** {fuel}")
            st.write(f"**Transmission:** {transmission}")
            st.write(f"**Owner:** {owner}")
            st.write(f"**Engine:** {engine_cc:,.0f} cc")
            st.write(f"**Power:** {max_power_bhp:.1f} BHP")

    # ========================================================
    # TAB 2 - MARKET INTELLIGENCE
    # ========================================================

    with tab2:

        st.subheader("📊 Market Intelligence")

        if benchmark_price is not None:

            asking_difference = (
                asking_price - benchmark_price
            )

            asking_difference_percentage = (
                asking_difference / benchmark_price
            ) * 100

            # -----------------------------------------------
            # Market status
            # -----------------------------------------------

            if asking_difference_percentage <= -10:

                market_status = "🟢 Potentially Underpriced"

            elif asking_difference_percentage >= 10:

                market_status = "🔴 Potentially Overpriced"

            else:

                market_status = "🟡 Fairly Priced"

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Seller Asking Price",
                    f"₹{asking_price:,.0f}"
                )

            with col2:

                st.metric(
                    "Market Benchmark",
                    f"₹{benchmark_price:,.0f}"
                )

            with col3:

                st.metric(
                    "Difference",
                    f"{asking_difference_percentage:+.2f}%"
                )

            st.divider()

            st.subheader("Market Price Classification")

            st.success(
                f"### {market_status}"
            )

            st.write(
                f"""
                The seller asking price is
                **{asking_difference_percentage:+.2f}%**
                compared with the comparable market benchmark.
                """
            )

            if benchmark_age is not None:

                st.info(
                    f"""
                    Comparable vehicles were matched using
                    **brand, fuel type, transmission and vehicle age**.
                    The closest benchmark age was
                    **{benchmark_age} years**.
                    """
                )

            st.write(
                f"Number of vehicles represented in the comparable "
                f"market group: **{comparable_count}**"
            )

        else:

            st.warning(
                """
                A comparable market benchmark could not be found
                for this combination of vehicle characteristics.
                """
            )

    # ========================================================
    # TAB 3 - MODEL INFORMATION
    # ========================================================

    with tab3:

        st.subheader("🤖 Machine Learning Model")

        st.write(
            """
            This application uses a trained machine learning regression
            model developed from used vehicle data.
            """
        )

        st.write(
            """
            The model considers vehicle characteristics such as
            vehicle age, kilometres driven, mileage, engine size,
            power, torque, seats, brand, model, fuel type,
            seller type, transmission and ownership.
            """
        )

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        if performance:

            st.subheader("📈 Model Performance")

            performance_items = {}

            for key, value in performance.items():

                if isinstance(value, (int, float)):

                    performance_items[key] = value

            if performance_items:

                performance_df = pd.DataFrame(
                    performance_items.items(),
                    columns=["Metric", "Value"]
                )

                st.dataframe(
                    performance_df,
                    use_container_width=True,
                    hide_index=True
                )

        st.divider()

        st.subheader("⚠️ Important Note")

        st.write(
            """
            The predicted price is an analytical estimate based on
            the training dataset and machine learning model. The
            market benchmark is based on comparable vehicles available
            in the dataset and should not be treated as a certified
            vehicle valuation.
            """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Used Vehicle Price Prediction & Market Intelligence System | "
    "Data Science Portfolio Project"
)
