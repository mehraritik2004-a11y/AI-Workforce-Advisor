# ================================================================
# AI Workforce Advisor - Streamlit Web App (Simplified Final Version)
# A Machine Learning-Based Decision Support System
# ================================================================

import pandas as pd
import joblib
import streamlit as st

# ----------------------------------------------------------------
# Page Setup
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AI Workforce Advisor",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------------------------------------------
# Load the saved model and encoder
# ----------------------------------------------------------------
# These files were created in Section 13 of our ML notebook.
model = joblib.load("best_model.pkl")
encoder = joblib.load("encoder.pkl")

# ----------------------------------------------------------------
# Home Page Section
# ----------------------------------------------------------------
st.title("🤖 AI Workforce Advisor")
st.subheader("A Machine Learning-Based Decision Support System for AI Productivity")

st.markdown("---")

st.markdown("""
### 📌 Problem Statement
Many professionals use AI tools every day, but most don't know whether they are
using them **effectively**. This app helps by:
- Predicting your expected **Productivity Gain** from AI usage
- Evaluating your **AI Adoption Level**
- Estimating your **Time Saved** (weekly & monthly)
- Giving you **personalized recommendations** to improve further
""")

st.markdown("---")
st.markdown("👈 **Use the sidebar to enter your details and click Predict.**")

# ----------------------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------------------
st.sidebar.header("📝 Enter Your Details")

industry = st.sidebar.selectbox(
    "Industry",
    ["Software Development", "Finance", "Healthcare", "Education",
     "Marketing", "Creative & Design"]
)

job_role = st.sidebar.selectbox(
    "Job Role",
    ["Software Engineer", "Data Scientist", "DevOps Engineer", "QA Tester",
     "Accountant", "Financial Analyst", "Investment Banker",
     "Clinical Admin", "Healthcare Analyst", "Medical Researcher",
     "Professor", "Teacher", "Student", "Instructional Designer",
     "Marketing Analyst", "SEO Specialist", "Social Media Manager",
     "Content Writer", "Graphic Designer", "Illustrator",
     "UI/UX Designer", "Video Editor"]
)

experience_years = st.sidebar.slider(
    "Experience (Years)",
    min_value=1, max_value=25, value=5
)

ai_tool = st.sidebar.selectbox(
    "Primary AI Tool",
    ["ChatGPT (OpenAI)", "Claude (Anthropic)", "Gemini (Google)",
     "GitHub Copilot", "Perplexity", "Midjourney", "DeepSeek"]
)

# ------------------------------------------------------------
# Friendly Usage Slider (replaces raw "Daily Token Usage" input)
# ------------------------------------------------------------
# Most people have no idea how many tokens they use per day, so instead
# of asking for a raw number, we ask a plain-language question and map
# the answer to a reasonable token estimate behind the scenes.
# This keeps the model's full accuracy (it still receives a numeric
# Daily_Token_Usage value) while being realistic for real users to answer.

usage_labels = {
    1: ("Rarely", 1000),
    2: ("Occasionally", 5000),
    3: ("Regularly", 15000),
    4: ("Heavily", 30000),
    5: ("Extremely heavily", 50000),
}

usage_choice = st.sidebar.select_slider(
    "How much do you use AI tools daily?",
    options=[1, 2, 3, 4, 5],
    value=3,
    format_func=lambda x: usage_labels[x][0]
)
daily_token_usage = usage_labels[usage_choice][1]  # numeric estimate fed to the model

tasks_automated = st.sidebar.slider(
    "Tasks Automated Per Week",
    min_value=1, max_value=12, value=3
)

predict_button = st.sidebar.button("🔮 Predict")

# ----------------------------------------------------------------
# Prediction Logic
# ----------------------------------------------------------------
if predict_button:

    # Step 1: Build a single-row DataFrame matching training columns
    user_input = pd.DataFrame([{
        "Industry": industry,
        "Job_Role": job_role,
        "Experience_Years": experience_years,
        "Primary_AI_Tool": ai_tool,
        "Daily_Token_Usage": daily_token_usage,
        "Tasks_Automated_Per_Week": tasks_automated
    }])

    # Step 2: Encode categorical columns using the same fitted encoder
    categorical_columns = ["Industry", "Job_Role", "Primary_AI_Tool"]
    numeric_columns = ["Experience_Years", "Daily_Token_Usage", "Tasks_Automated_Per_Week"]

    encoded_array = encoder.transform(user_input[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns)

    final_input = pd.concat([user_input[numeric_columns], encoded_df], axis=1)

    # Step 3: Predict
    predicted_gain = model.predict(final_input)[0]
    predicted_gain = max(0, predicted_gain)

    # ------------------------------------------------------------
    # Display Results (clean, no charts)
    # ------------------------------------------------------------
    st.markdown("## 📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Productivity Gain", f"{predicted_gain:.2f}%")
        if predicted_gain < 10:
            level, color = "Needs Improvement", "🔴"
        elif predicted_gain < 20:
            level, color = "Average", "🟠"
        elif predicted_gain < 35:
            level, color = "Good", "🟡"
        else:
            level, color = "Excellent", "🟢"
        st.markdown(f"{color} **{level}**")

    weekly_hours_saved = (tasks_automated * 30) / 60
    monthly_hours_saved = weekly_hours_saved * 4

    with col2:
        st.metric("Weekly Time Saved", f"{weekly_hours_saved:.1f} hrs")
        st.metric("Monthly Time Saved", f"{monthly_hours_saved:.1f} hrs")

    # ------------------------------------------------------------
    # AI Adoption Score (out of 100)
    # ------------------------------------------------------------
    # Weight distribution:
    #   Daily Usage Level      -> 30 marks
    #   Tasks Automated        -> 25 marks
    #   Productivity Gain      -> 25 marks
    #   Experience             -> 10 marks
    #   AI Tool                -> 10 marks
    usage_score = (usage_choice / 5) * 30

    tasks_score = min(tasks_automated / 12, 1) * 25
    productivity_score = min(predicted_gain / 50, 1) * 25
    experience_score = min(experience_years / 25, 1) * 10

    tool_score_map = {
        "ChatGPT (OpenAI)": 10, "Claude (Anthropic)": 10, "Gemini (Google)": 9,
        "GitHub Copilot": 9, "Perplexity": 8, "DeepSeek": 8, "Midjourney": 7
    }
    tool_score = tool_score_map[ai_tool]

    adoption_score = round(usage_score + tasks_score + productivity_score + experience_score + tool_score, 1)

    if adoption_score < 40:
        adoption_level = "Low"
    elif adoption_score < 70:
        adoption_level = "Medium"
    else:
        adoption_level = "High"

    with col3:
        st.metric("AI Adoption Score", f"{adoption_score} / 100")
        st.markdown(f"**{adoption_level} adoption**")

    st.markdown("---")

    # ------------------------------------------------------------
    # Personalized Recommendations
    # ------------------------------------------------------------
    recommendations = []

    if usage_choice <= 2:
        recommendations.append("📈 Your AI usage is low. Try using AI tools more regularly in your daily workflow.")

    if tasks_automated < 3:
        recommendations.append("🔁 Few tasks are automated. Identify repetitive tasks in your work and automate them with AI.")

    if predicted_gain < 10:
        recommendations.append("📝 Your productivity gain is low. Try using AI for documentation, summarizing, or drafting work.")

    if experience_years < 5:
        recommendations.append("🎓 You're early in your career. Learning Prompt Engineering can help you get much more out of AI tools.")

    if len(recommendations) == 0:
        recommendations.append("✅ Great job! You're using AI effectively across usage, automation, and experience.")

    st.markdown("### 💡 Personalized Recommendations")
    for tip in recommendations:
        st.write("- " + tip)

    # One-line insight instead of a chart (from feature importance analysis)
    st.info(
        "ℹ️ Note: your usage habits (how often you use AI, how many tasks you automate) "
        "matter far more to this prediction than your industry, role, or AI tool choice."
    )

    # ------------------------------------------------------------
    # Overall Assessment (Summary Paragraph)
    # ------------------------------------------------------------
    summary_text = (
        f"Based on your profile, your predicted **Productivity Gain** is "
        f"**{predicted_gain:.2f}%**, which falls under the **{level}** category. "
        f"Your overall **AI Adoption Score** is **{adoption_score}/100** "
        f"({adoption_level} adoption level), and you are estimated to save around "
        f"**{weekly_hours_saved:.1f} hours per week** (**{monthly_hours_saved:.1f} hours per month**) "
        f"through AI-driven task automation."
    )

    st.markdown("### 🧾 Overall Assessment")
    st.info(summary_text)
