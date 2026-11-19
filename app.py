import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# CUSTOM DARK THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1b, #111a32, #21153b);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111a30, #24143c);
    border-right: 1px solid #475569;
}

h1, h2, h3 {
    color: #e0e7ff !important;
}

p, label {
    color: #dbeafe !important;
}

.hero {
    background: linear-gradient(110deg, #312e81, #6d28d9, #075985);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid #818cf8;
}

.hero h1 {
    color: white !important;
    font-size: 34px;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 17px;
}

.metric-card {
    background: linear-gradient(145deg, #17233d, #29204b);
    border: 1px solid #475569;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.metric-title {
    color: #a5b4fc;
    font-size: 14px;
}

.metric-value {
    color: #67e8f9;
    font-size: 27px;
    font-weight: bold;
}

.stButton button {
    background: linear-gradient(90deg, #6366f1, #9333ea);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton button:hover {
    background: linear-gradient(90deg, #4f46e5, #7e22ce);
    color: white;
}

.section-card {
    background: #17233d;
    border: 1px solid #475569;
    padding: 20px;
    border-radius: 15px;
}

[data-testid="stMetric"] {
    background: #17233d;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# DATASET
# ==========================================

FEATURES = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score",
    "Sleep_Hours"
]

TARGET = "Final_Marks"


def generate_dataset():

    np.random.seed(42)

    n = 60

    study_hours = np.random.uniform(1, 10, n)
    attendance = np.random.uniform(50, 100, n)
    previous_marks = np.random.uniform(30, 95, n)
    assignment_score = np.random.uniform(35, 100, n)
    sleep_hours = np.random.uniform(4, 9, n)

    noise = np.random.normal(0, 7, n)

    final_marks = (
        0.25 * study_hours * 10
        + 0.20 * attendance
        + 0.25 * previous_marks
        + 0.25 * assignment_score
        + 0.10 * sleep_hours * 10
        + noise
    )

    final_marks = np.clip(final_marks, 0, 100)

    return pd.DataFrame({
        "Study_Hours": study_hours.round(1),
        "Attendance": attendance.round(1),
        "Previous_Marks": previous_marks.round(1),
        "Assignment_Score": assignment_score.round(1),
        "Sleep_Hours": sleep_hours.round(1),
        "Final_Marks": final_marks.round(1)
    })


# ==========================================
# SESSION STATE
# ==========================================

if "df" not in st.session_state:
    st.session_state.df = generate_dataset()

if "model" not in st.session_state:
    st.session_state.model = None

if "metrics" not in st.session_state:
    st.session_state.metrics = None

if "importance" not in st.session_state:
    st.session_state.importance = None

if "test_data" not in st.session_state:
    st.session_state.test_data = None


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def metric_card(title, value):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def performance_category(marks):

    if marks >= 75:
        return "High Performance"

    elif marks >= 50:
        return "Average Performance"

    else:
        return "Low Performance"


def train_model(df):

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return model, metrics, importance, (y_test, predictions)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("# 🎓 Student AI")

st.sidebar.caption("Machine Learning Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📊 Dataset & EDA",
        "🤖 Model Training",
        "📈 Model Results",
        "🔮 Predict Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Project Information")

st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write("**Model:** Regression")
st.sidebar.write("**Dataset:** 60 Students")
st.sidebar.write("**Language:** Python")

st.sidebar.markdown("---")

st.sidebar.caption("PBL Project | 2026-27")


# ==========================================
# 1. DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🎓 Student Performance Prediction</h1>

        <p>
        An intelligent Machine Learning system
        that analyses student academic factors
        and predicts their performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Students", len(df))

    with c2:
        metric_card(
            "Average Marks",
            f"{df['Final_Marks'].mean():.2f}"
        )

    with c3:
        metric_card(
            "Average Attendance",
            f"{df['Attendance'].mean():.1f}%"
        )

    with c4:
        metric_card(
            "Average Study Hours",
            f"{df['Study_Hours'].mean():.1f} hrs"
        )

    st.markdown("## 📌 Project Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="section-card">

        <h3>🎯 Project Objective</h3>

        <p>
        To predict student academic performance
        using Machine Learning by analysing
        attendance, study hours, previous marks,
        assignment scores and sleep hours.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="section-card">

        <h3>⚙️ Technologies Used</h3>

        <p>
        Python, Pandas, NumPy, Plotly,
        Scikit-learn and Streamlit.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🔄 Machine Learning Workflow")

    st.info(
        "Dataset → Preprocessing → EDA → "
        "Train-Test Split → Random Forest → "
        "Evaluation → Prediction"
    )

    st.markdown("## 📊 Marks Distribution")

    fig = px.histogram(
        df,
        x="Final_Marks",
        nbins=12,
        color_discrete_sequence=["#818cf8"],
        title="Student Final Marks Distribution"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 2. DATASET AND EDA
# ==========================================

elif page == "📊 Dataset & EDA":

    st.title("📊 Dataset & Exploratory Data Analysis")

    uploaded_file = st.file_uploader(
        "Upload Student Dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            if set(FEATURES + [TARGET]).issubset(
                uploaded_df.columns
            ):

                uploaded_df = uploaded_df[
                    FEATURES + [TARGET]
                ].apply(pd.to_numeric, errors="coerce")

                uploaded_df = uploaded_df.dropna()

                if uploaded_df.empty:
                    st.error("The uploaded dataset has no valid records.")

                else:
                    st.session_state.df = uploaded_df
                    st.session_state.model = None
                    st.session_state.metrics = None
                    st.session_state.importance = None
                    st.session_state.test_data = None

                    st.success("Dataset uploaded successfully!")

            else:

                st.error(
                    "Required columns: " +
                    ", ".join(FEATURES + [TARGET])
                )

        except Exception as e:

            st.error(f"Error reading dataset: {e}")

    df = st.session_state.df

    st.markdown("### Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.write("Dataset Shape:", df.shape)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Study Hours vs Marks")

        fig = px.scatter(
            df,
            x="Study_Hours",
            y="Final_Marks",
            color="Attendance",
            color_continuous_scale="viridis"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.markdown("### Attendance vs Marks")

        fig = px.scatter(
            df,
            x="Attendance",
            y="Final_Marks",
            color="Previous_Marks",
            color_continuous_scale="plasma"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### 🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Statistical Summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )


# ==========================================
# 3. MODEL TRAINING
# ==========================================

elif page == "🤖 Model Training":

    st.title("🤖 Random Forest Model Training")

    st.write(
        "Train a Random Forest Regression model "
        "using the student dataset."
    )

    st.markdown("### Model Configuration")

    st.info(
        "Algorithm: Random Forest Regression\n\n"
        "Training Split: 80%\n\n"
        "Testing Split: 20%\n\n"
        "Number of Estimators: 100"
    )

    if st.button("🚀 Train Machine Learning Model"):

        with st.spinner("Training model..."):

            try:

                model, metrics, importance, test_data = train_model(
                    st.session_state.df
                )

                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.importance = importance
                st.session_state.test_data = test_data

                st.success("Model trained successfully!")

            except Exception as e:

                st.error(f"Training error: {e}")

    if st.session_state.model is not None:

        st.markdown("## Training Results")

        metrics = st.session_state.metrics

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("MAE", f"{metrics['MAE']:.2f}")

        with c2:
            metric_card("MSE", f"{metrics['MSE']:.2f}")

        with c3:
            metric_card("RMSE", f"{metrics['RMSE']:.2f}")

        with c4:
            metric_card("R² Score", f"{metrics['R2']:.2f}")

    else:

        st.warning("Click the training button to train the model.")


# ==========================================
# 4. MODEL RESULTS
# ==========================================

elif page == "📈 Model Results":

    st.title("📈 Model Evaluation & Results")

    if st.session_state.model is None:

        st.warning("Please train the model first.")

    else:

        metrics = st.session_state.metrics

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("MAE", f"{metrics['MAE']:.2f}")

        with c2:
            metric_card("MSE", f"{metrics['MSE']:.2f}")

        with c3:
            metric_card("RMSE", f"{metrics['RMSE']:.2f}")

        with c4:
            metric_card("R² Score", f"{metrics['R2']:.2f}")

        y_test, predictions = st.session_state.test_data

        st.markdown("## Actual vs Predicted Marks")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_test))),
                y=y_test.values,
                mode="lines+markers",
                name="Actual Marks"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(predictions))),
                y=predictions,
                mode="lines+markers",
                name="Predicted Marks"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title="Actual vs Predicted Performance",
            xaxis_title="Test Student",
            yaxis_title="Final Marks"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("## Feature Importance")

        importance = st.session_state.importance

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="viridis"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        comparison = pd.DataFrame({
            "Actual Marks": y_test.values,
            "Predicted Marks": predictions
        })

        st.markdown("## Prediction Comparison")

        st.dataframe(
            comparison.round(2),
            use_container_width=True
        )


# ==========================================
# 5. STUDENT PERFORMANCE PREDICTION
# ==========================================

elif page == "🔮 Predict Performance":

    st.title("🔮 Predict Student Performance")

    st.write(
        "Enter student academic details to "
        "predict their final examination marks."
    )

    if st.session_state.model is None:

        st.warning(
            "Please go to Model Training and train the model first."
        )

    else:

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                study_hours = st.slider(
                    "Daily Study Hours",
                    0.0, 12.0, 5.0, 0.5
                )

                attendance = st.slider(
                    "Attendance Percentage",
                    0.0, 100.0, 80.0, 1.0
                )

                previous_marks = st.slider(
                    "Previous Academic Marks",
                    0.0, 100.0, 70.0, 1.0
                )

            with col2:

                assignment_score = st.slider(
                    "Assignment Score",
                    0.0, 100.0, 75.0, 1.0
                )

                sleep_hours = st.slider(
                    "Daily Sleep Hours",
                    0.0, 12.0, 7.0, 0.5
                )

            submit = st.form_submit_button(
                "✨ Predict Performance"
            )

        if submit:

            input_data = pd.DataFrame([{
                "Study_Hours": study_hours,
                "Attendance": attendance,
                "Previous_Marks": previous_marks,
                "Assignment_Score": assignment_score,
                "Sleep_Hours": sleep_hours
            }])

            prediction = st.session_state.model.predict(
                input_data
            )[0]

            prediction = float(np.clip(prediction, 0, 100))

            category = performance_category(prediction)

            st.markdown("---")

            st.markdown("## 🎯 Prediction Result")

            c1, c2 = st.columns(2)

            with c1:

                metric_card(
                    "Predicted Final Marks",
                    f"{prediction:.2f}/100"
                )

            with c2:

                metric_card(
                    "Performance Category",
                    category
                )

            if prediction >= 75:

                st.success(
                    "Excellent! The model predicts high performance."
                )

            elif prediction >= 50:

                st.warning(
                    "The model predicts average performance."
                )

            else:

                st.error(
                    "The model predicts low performance."
                )

            st.progress(int(prediction))

            st.caption(
                "This is an estimated prediction, "
                "not a guaranteed examination result."
            )


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<center style="color:#94a3b8;">

Student Performance Prediction Using Machine Learning

<br>

Developed using Python | Streamlit | Scikit-learn

</center>
""", unsafe_allow_html=True)
