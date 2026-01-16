import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
import matplotlib
matplotlib.use("Agg")

# Streamlit Page Config
st.set_page_config(page_title="Loan approval prediction", layout="centered")

education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"Yes": 1, "No": 0}


if "page" not in st.session_state:
    st.session_state.page = "Home"

try:
    params = st.query_params
    if "page" in params:
        st.session_state.page = params["page"]
except:
    pass


def generate_pdf(dataframe, approved, risk):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Header background
    c.setFillColorRGB(0.08, 0.72, 0.65)
    c.rect(0, height - 80, width, 80, fill=1, stroke=0)

    # Title
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 50, "Loan Prediction Report")

    c.setFillColorRGB(0, 0, 0)

    # Loan Status
    c.setFont("Helvetica-Bold", 14)
    status_text = "LOAN APPROVED " if approved else "LOAN REJECTED "
    c.drawString(50, height - 120, status_text)

    # Risk level
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 145, f"Risk Level: {risk}")

    # Divider
    c.line(50, height - 155, width - 50, height - 155)

    # Applicant Details
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 185, "Applicant Details")

    y = height - 215
    for col, val in dataframe.iloc[0].items():
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y, f"{col}:")
        c.setFont("Helvetica", 11)
        c.drawString(220, y, str(val))
        y -= 20

    # Positive Factors
    y -= 15
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Positive Factors")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(60, y, "- Good credit score")
    y -= 15
    c.drawString(60, y, "- Stable income")
    y -= 15
    c.drawString(60, y, "- Asset backing")

    # Areas for Improvement
    y -= 25
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Areas for Improvement")
    y -= 18

    c.setFont("Helvetica", 11)
    if approved:
        c.drawString(60, y, "No significant concerns identified.")
    else:
        c.drawString(60, y, "- Improve credit score")
        y -= 15
        c.drawString(60, y, "- Reduce loan amount")
        y -= 15
        c.drawString(60, y, "- Increase income stability")

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawString(
        50,
        40,
        "This report is system generated based on the applicant's submitted information."
    )

    c.save()
    buffer.seek(0)
    return buffer


# Load Model & Dataset
df = pd.read_csv("data/loan_data.csv") 
MODEL_PATH = os.path.join("models", "loan_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

try:
    expected_features = model.feature_names_in_
except:
    expected_features = df.drop("loan_status", axis=1).columns
                    
# Global Styling

st.markdown("""
<style>
.block-container {
    max-width: 1400px !important; 
    padding-top: 2rem;
    padding-bottom: 2rem;
    margin: auto;
}

/* Background */
body {
    background-color: #f5f7fb;
}

/* Cards */
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 16px;
    margin-top: 20px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
}

/* Metrics */
.metric-grid {
    display: flex;
    gap: 16px;
    margin-top: 20px;
}
.metric-box {
    flex: 1;
    background: #ffffff;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #14b8a6, #06b6d4);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0d9488, #0284c7);
}

.good { color: #16a34a; font-weight: bold; }
.bad { color: #dc2626; font-weight: bold; }
            
.status-approved {
    background: #E6FFF3;
    border-left: 6px solid #22C55E;
    padding: 22px;
    border-radius: 14px;
}

.status-rejected {
    background: #FFECEC;
    border-left: 6px solid #EF4444;
    padding: 22px;
    border-radius: 14px;
}

.status-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 6px;
}

.status-sub {
    color: #374151;
    font-size: 14px;
}
        
.button-row {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-top: 25px;
}

.button-row .stButton > button,
.button-row .stDownloadButton > button {
    width: 220px;
    height: 45px;
}

.action-btn > button {
    width: 200px !important;
    height: 45px !important;
}

.stColumns {
    gap: 1px !important;
}

i {
    margin-right: 6px;
    vertical-align: middle;
}
        
/* HERO STRIP */
.hero-strip {
    background: linear-gradient(90deg, #06b6d4, #0ea5e9);
    padding: 22px 26px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
}

.hero-strip h2 {
    margin: 0;
    font-size: 26px;
    font-weight: 700;
}

.hero-strip p {
    margin-top: 6px;
    font-size: 14px;
    opacity: 0.95;
}

/* FEATURE GRID */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 18px;
}

.feature-card {
    background: #f8fafc;
    padding: 14px 16px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
}

.feature-card i {
    font-size: 18px;
    color: #0ea5e9;
}

.home-card {
    width: 100%;          
    max-width: 1200px;    
    min-height: 620px; 
    margin: 40px auto;
    padding: 48px 52px;  
    border-radius: 20px;
    position: relative; 
}

.home-cta {
    display: flex;
    justify-content: center;
    margin-top: 40px;
}

.home-cta a{
    background: linear-gradient(90deg, #06b6d4, #0ea5e9);
    color: white;
    padding: 12px 36px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 500;
    text-decoration: none;
}

.home-cta a:hover {
    background: linear-gradient(90deg, #0d9488, #0284c7);
}

</style>
""",  unsafe_allow_html=True)

st.markdown("""
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
""", unsafe_allow_html=True)


# PAGE NAVIGATION (SIDEBAR)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Loan Prediction", "Dashboard", "Insights"],
    index=["Home", "Loan Prediction", "Dashboard", "Insights"].index(st.session_state.page)
)
st.session_state.page = page


# HOME PAGE
if page == "Home":
    st.markdown("""
    <div class="card home-card">
        <div class="hero-strip">
            <h2>
                <i class="fa-solid fa-building-columns"></i>
                Loan Approval Prediction System
            </h2>
            <p>
                ML-powered loan eligibility analysis using
                Machine Learning & Streamlit
            </p>
        </div>
        <h4>What this application offers</h4>
        <div class="feature-grid">
            <div class="feature-card">
                <i class="fa-solid fa-circle-check"></i>
                Predict loan approval status
            </div>
            <div class="feature-card">
                <i class="fa-solid fa-chart-pie"></i>
                Visualize dataset insights
            </div>
            <div class="feature-card">
                <i class="fa-solid fa-gauge-high"></i>
                Analyze approval & rejection metrics
            </div>
            <div class="feature-card">
                <i class="fa-solid fa-chart-line"></i>
                Explore interactive dashboards
            </div>
        </div>
        <div class="home-cta">
            <a href="?page=Loan Prediction">Start Application</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


# LOAN PREDICTION PAGE

elif page == "Loan Prediction":
    st.markdown("""
    <h2><i class="fa-solid fa-building-columns"></i> Loan Application</h2>
    """, unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>ML-powered loan eligibility analysis</p>", unsafe_allow_html=True)

# FORM
    with st.form("loan_form"):

        st.markdown("""
        <h4><i class="fa-solid fa-user"></i> Personal Information</h4>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            no_of_dependents = st.number_input("Dependents", 0, 10, 0)

        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            
        st.markdown("""
        <h4><i class="fa-solid fa-wallet"></i> Financial Information</h4>
        """, unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            income_annum = st.number_input("Annual Income", min_value=0, value=30000, step=10000)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700,step=10)

        with col4:
            loan_amount = st.number_input("Loan Amount", min_value=0, value=500000, step=50000)
            loan_term = st.number_input("Loan Term (Months)", min_value=1, max_value=480, value=12, step=6)

        st.markdown("""
        <h4><i class="fa-solid fa-house"></i> Assets Information</h4>
        """, unsafe_allow_html=True)

        col5, col6 = st.columns(2)
        with col5:
            residential_assets_value = st.number_input("Residential Assets Value",min_value=0, value=0)
            commercial_assets_value = st.number_input("Commercial Assets Value",min_value=0, value=0)

        with col6:
            luxury_assets_value = st.number_input("Luxury Assets Value",min_value=0, value=0)
            bank_asset_value = st.number_input("Bank Assets Value",min_value=0, value=0)

        submitted = st.form_submit_button("Predict Loan Approval")



    # RESULTS
    if submitted:
        input_df = pd.DataFrame([{
            "education": education_map[education],
            "self_employed": self_employed_map[self_employed],
            "no_of_dependents": no_of_dependents,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }])
        input_df = input_df.reindex(columns=expected_features, fill_value=0)
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        approved_idx = classes.index("Approved") if "Approved" in classes else 1
        approved_prob = proba[approved_idx] * 100
        rejection_prob = 100 - approved_prob
        approved = approved_prob >= 0.70
        if approved_prob >= 80:
            risk = "Low"
        elif approved_prob >= 60:
            risk = "Medium"
        else:
            risk = "High"

        monthly_payment = loan_amount / loan_term
        interest_rate = 9.5 if approved else 11.5

         # Extreme edge cases handling

        if ( income_annum < 50000 and no_of_dependents >= 5 and loan_amount > 10 * income_annum ):
            approved = False
            approved_prob = min(approved_prob, 35)
            rejection_prob = 100 - approved_prob
            risk = "High"

        # Result card
        st.markdown(f"""
        <div class="{'status-approved' if approved else 'status-rejected'}">
            <div class="status-title">
                {"Loan Approved" if approved else "Loan Rejected"}
            </div>
            <div class="status-sub">
                {"The applicant satisfies the eligibility criteria."
                if approved else
                "The applicant does not meet the eligibility criteria."}
            </div>
            <div style="display:flex; gap:40px; margin-top:18px;">
                <div>
                    <div style="font-size:13px; color:#6b7280;">Approval Probability</div>
                    <div style="font-size:22px; font-weight:600;">{approved_prob:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:13px; color:#6b7280;">Rejection Probability</div>
                    <div style="font-size:22px; font-weight:600;">{rejection_prob:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # PROBABILITY BREAKDOWN
        st.markdown("""
        <h4 style="margin-top:30px;">Probability Breakdown</h4>
        """, unsafe_allow_html=True)

        st.progress(approved_prob / 100)
        st.caption("Approval Confidence")

        # METRICS
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-box">
                <h4><i class="fa-solid fa-shield-halved"></i> Risk Level</h4>
                <p class="{'good' if risk=='Low' else 'bad'}">{risk}</p>
            </div>
            <div class="metric-box">
                <h4><i class="fa-solid fa-indian-rupee-sign"></i> Monthly Payment</h4>
                <p>₹ {monthly_payment:,.2f}</p>
            </div>
            <div class="metric-box">
                <h4><i class="fa-solid fa-percent"></i> Interest Rate</h4>
                <p>{interest_rate}%</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # INSIGHTS
        st.markdown("""
        <h4 class="good">
                    <i class="fa-solid fa-circle-check"></i> Positive Factors
        </h4>
        <ul>
            <li>Good credit score</li>
            <li>Stable income</li>
            <li>Asset backing</li>
        </ul>
        """, unsafe_allow_html=True)

        # AREAS FOR IMPROVEMENT (CONDITIONAL)
        if approved:
            st.markdown("""
            <h4 class="good">
                        <i class="fa-solid fa-thumbs-up"></i> Areas for Improvement
            </h4>
            <p style="color:#16a34a;">No significant concerns identified.</p>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <h4 class="bad">
                        <i class="fa-solid fa-triangle-exclamation"></i> Areas for Improvement
            </h4>
            <ul>
                <li>High existing debt burden</li>
                <li>Loan amount exceeds recommended limits</li>
                <li>High loan-to-value ratio</li>
            </ul>
            """, unsafe_allow_html=True)
            
        # ACTION BUTTONS 
        left, center1, center2, right = st.columns([2, 2, 2, 2])

        with center1:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("Start New Application"):
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with center2:
            pdf_file = generate_pdf(input_df, approved, risk)
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            st.download_button(
                "Download Report",
                data=pdf_file,
                file_name="loan_report.pdf",
                mime="application/pdf"
            )
            st.markdown('</div>', unsafe_allow_html=True)

# DASHBOARD PAGE

elif page == "Dashboard":
    st.markdown("""
    <h2>
    <i class="fa-solid fa-chart-line"></i> Dashboard Overview
    </h2>
    """, unsafe_allow_html=True)

    total = len(df)
    df.columns = df.columns.str.strip()
    df["loan_status"] = df["loan_status"].str.strip().str.capitalize()
    approved = len(df[df["loan_status"] == "Approved"])
    rejected = len(df[df["loan_status"] == "Rejected"])
    approval_rate = round((approved / total) * 100, 2) 
    rejection_rate = round((rejected / total) * 100, 2) 

    st.write("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applications", total)
    col2.metric("Approval Rate", f"{approval_rate}%")
    col3.metric("Rejection Rate", f"{rejection_rate}%")

    st.write("---")
    st.write("### Income Distribution")
    fig, ax = plt.subplots(figsize=(5,3.5), dpi=100)
    df.columns = df.columns.str.strip()
    sns.histplot(df["income_annum"], ax=ax)
    plt.tight_layout()
    st.pyplot(fig, width='content')


# INSIGHTS PAGE (VISUAL GRAPHS)

elif page == "Insights":
    df.columns = df.columns.str.strip()
    st.markdown("""
    <h2>
    <i class="fa-solid fa-chart-column"></i> Visual Insights
    </h2>
    """, unsafe_allow_html=True)


    # Graph 1
    st.write("### Loan Approval Distribution")
    fig1, ax1 = plt.subplots(figsize=(5,3.5), dpi=120)
    sns.countplot(x="loan_status", data=df, ax=ax1)
    plt.tight_layout()
    st.pyplot(fig1, width='content')

    st.write("---")
    # Graph 2
    st.write("### Applicant Income vs Loan Amount")
    fig2, ax2 = plt.subplots(figsize=(5,3.5), dpi=120)
    sns.scatterplot(
        x="income_annum",
        y="loan_amount",
        hue="loan_status",
        data=df,
        ax=ax2
    )
    plt.tight_layout()
    st.pyplot(fig2, width='content')

    st.write("---")
    # Graph 3
    st.write("### Education-wise Loan Approval")
    fig3, ax3 = plt.subplots(figsize=(5,3.5), dpi=120)
    sns.countplot(x="education", hue="loan_status", data=df, ax=ax3)
    plt.tight_layout()
    st.pyplot(fig3, width='content')

