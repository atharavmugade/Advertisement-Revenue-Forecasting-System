import streamlit as st
import pandas as pd
import pickle

# ------------------ PAGE TITLE ------------------
st.set_page_config(page_title="Advertisement Revenue Forecasting System", page_icon="📊")

st.title("📊 Advertisement Revenue Forecasting System")
st.markdown("### Predict Approved Advertisement Conversion using Machine Learning")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("cleaned-data.csv")
model = pickle.load(open("model.pkl", "rb"))

# ------------------ SIDEBAR ------------------
st.sidebar.header("Enter Advertisement Details")

ad_id = st.sidebar.number_input("Ad ID", min_value=100000, step=1)

xyz_campaign_id = st.sidebar.selectbox(
    "XYZ Campaign ID",
    sorted(df['xyz_campaign_id'].unique())
)

fb_campaign_id = st.sidebar.selectbox(
    "FB Campaign ID",
    sorted(df['fb_campaign_id'].unique())
)

age = st.sidebar.selectbox(
    "Age Group",
    sorted(df['age'].unique())
)

gender = st.sidebar.selectbox(
    "Gender",
    sorted(df['gender'].unique())
)

interest = st.sidebar.selectbox(
    "Interest",
    sorted(df['interest'].unique())
)

impressions = st.sidebar.number_input(
    "Impressions",
    min_value=0,
    step=100
)

clicks = st.sidebar.number_input(
    "Clicks",
    min_value=0,
    step=1
)

spent = st.sidebar.number_input(
    "Amount Spent",
    min_value=0.0,
    step=0.10
)

total_conversion = st.sidebar.number_input(
    "Total Conversion",
    min_value=0,
    step=1
)

# ------------------ PREDICTION ------------------
if st.sidebar.button("🚀 Predict"):

    st.subheader("📋 Entered Details")

    st.write("**Ad ID:**", ad_id)
    st.write("**XYZ Campaign ID:**", xyz_campaign_id)
    st.write("**FB Campaign ID:**", fb_campaign_id)
    st.write("**Age Group:**", age)
    st.write("**Gender:**", gender)
    st.write("**Interest:**", interest)
    st.write("**Impressions:**", impressions)
    st.write("**Clicks:**", clicks)
    st.write("**Amount Spent:**", spent)
    st.write("**Total Conversion:**", total_conversion)

    columns = [
        'ad_id',
        'xyz_campaign_id',
        'fb_campaign_id',
        'age',
        'gender',
        'interest',
        'Impressions',
        'Clicks',
        'Spent',
        'Total_Conversion'
    ]

    myinput = [[
        ad_id,
        xyz_campaign_id,
        fb_campaign_id,
        age,
        gender,
        interest,
        impressions,
        clicks,
        spent,
        total_conversion
      ]]

    myinput = pd.DataFrame(myinput, columns=columns)

    result = model.predict(myinput)

    prediction = float(result[0])

    # Convert prediction into percentage
    percentage = round(prediction * 10, 2)

    if percentage > 100:
        percentage = 100

    st.markdown("---")

    if prediction < 0:
        st.error("❌ Invalid Inputs")
    else:
        st.balloons()

        st.success("✅ Prediction Completed Successfully")

        st.metric(
            label="Advertisement Success Rate",
            value=f"{percentage}%"
        )
# ------------------ DATASET INFO ------------------
st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("📁 Total Records", len(df))
col2.metric("📢 Campaigns", df['xyz_campaign_id'].nunique())
col3.metric("👥 Age Groups", df['age'].nunique())

# ------------------ ABOUT ------------------
with st.expander("ℹ About Project"):
    st.write("""
This application predicts **Approved Advertisement Conversion**
using a Machine Learning model.

### Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-Learn
- Random Forest Regression
""")

st.markdown("---")
st.caption("Made with ❤️ by Atharav_Mugade")
