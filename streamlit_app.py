import streamlit as st
import numpy as np
import joblib
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_extras.colored_header import colored_header
import time

# Page Config
st.set_page_config(
    page_title="AI Exam Score Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: transparent;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .input-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .score-display {
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .stNumberInput > div > div > input {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .tip-card {
        background: white;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    .progress-ring {
        transform: rotate(-90deg);
    }
    
    .progress-ring-circle {
        stroke: #667eea;
        stroke-width: 8;
        fill: transparent;
        stroke-dasharray: 377;
        stroke-dashoffset: 377;
        transition: stroke-dashoffset 0.5s ease-in-out;
    }
    
    .section-header {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.3rem;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .time-allocation {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .sidebar .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    @media (max-width: 768px) {
        .main-container {
            margin: 0.5rem;
            padding: 1rem;
        }
        
        .score-display {
            font-size: 2.5rem;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .tooltip {
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load("linear_regression_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.info("Please ensure 'linear_model.pkl' and 'scaler.pkl' are in the same directory")
        st.stop()

model, scaler = load_models()

# Initialize session state
if "study_hours" not in st.session_state:
    st.session_state.study_hours = 4.0
if "sleep_hours" not in st.session_state:
    st.session_state.sleep_hours = 7.0
if "social_media_hours" not in st.session_state:
    st.session_state.social_media_hours = 2.0
if "netflix_hours" not in st.session_state:
    st.session_state.netflix_hours = 1.5
if "history" not in st.session_state:
    st.session_state.history = []
if "show_insights" not in st.session_state:
    st.session_state.show_insights = False

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Quick Stats")
    
    if st.session_state.history:
        avg_score = np.mean([h['score'] for h in st.session_state.history])
        st.metric("Average Score", f"{avg_score:.1f}")
        st.metric("Predictions Made", len(st.session_state.history))
    else:
        st.info("Make your first prediction to see stats!")
    
    st.markdown("---")
    
    # Theme selector
    st.markdown("### 🎨 Personalization")
    study_goal = st.selectbox(
        "Study Goal",
        ["Excellent (90+)", "Good (70-89)", "Pass (50-69)", "Improvement"]
    )
    
    show_tips = st.checkbox("Show Performance Tips", value=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content in container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header with animation
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #2d3748; font-size: 3rem; font-weight: 700; margin: 0;">
        🎯 AI Exam Score Predictor
    </h1>
    <p style="color: #718096; font-size: 1.2rem; margin: 0.5rem 0;">
        Optimize your study routine with AI-powered insights
    </p>
</div>
""", unsafe_allow_html=True)

# Input sections
st.markdown('<div class="section-header">📚 Study & Life Balance</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("#### 📖 Academic Activities")
    
    st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=16.0,
        step=0.5,
        key="study_hours",
        help="Focused study time including reading, assignments, and review"
    )
    
    attendance_percentage = st.slider(
        "Class Attendance (%)", 
        0, 100, 85,
        help="Regular attendance strongly correlates with better performance"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("#### 🏃 Health & Wellness")
    
    st.number_input(
        "Sleep Hours per Day",
        min_value=4.0,
        max_value=12.0,
        step=0.5,
        key="sleep_hours",
        help="Quality sleep is crucial for memory consolidation"
    )
    
    exercise_frequency = st.slider(
        "Exercise Sessions per Week", 
        0, 14, 3,
        help="Regular exercise improves cognitive function and stress management"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Entertainment and distractions
st.markdown('<div class="section-header">📱 Digital Activities</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.number_input(
        "Social Media Hours",
        min_value=0.0,
        max_value=12.0,
        step=0.5,
        key="social_media_hours",
        help="Time spent on social platforms daily"
    )

with col4:
    st.number_input(
        "Entertainment Hours",
        min_value=0.0,
        max_value=8.0,
        step=0.5,
        key="netflix_hours",
        help="Netflix, gaming, and other entertainment"
    )

# Mental health rating
st.markdown('<div class="section-header">🧘 Mental Wellbeing</div>', unsafe_allow_html=True)

mental_health_rating = st.slider(
    "Mental Health Rating (1-10)", 
    1, 10, 6,
    help="Rate your current stress levels, motivation, and overall mental wellbeing"
)

# Time allocation with circular progress
used_hours = (
    st.session_state.study_hours +
    st.session_state.sleep_hours +
    st.session_state.social_media_hours +
    st.session_state.netflix_hours
)
remaining_hours = 24.0 - used_hours

st.markdown('<div class="section-header">⏰ Time Allocation Analysis</div>', unsafe_allow_html=True)

col5, col6, col7 = st.columns([2, 1, 1])

with col5:
    # Create a donut chart for time allocation
    labels = ['Study', 'Sleep', 'Social Media', 'Entertainment', 'Other']
    values = [
        st.session_state.study_hours,
        st.session_state.sleep_hours,
        st.session_state.social_media_hours,
        st.session_state.netflix_hours,
        max(0, remaining_hours)
    ]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#a8edea'])
    )])
    
    fig.update_layout(
        title="Daily Time Distribution",
        height=400,
        showlegend=True,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0;">Total Hours</h3>
        <div style="font-size: 2rem; font-weight: 700;">{used_hours:.1f}/24</div>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0;">Remaining</h3>
        <div style="font-size: 2rem; font-weight: 700;">{remaining_hours:.1f}h</div>
    </div>
    """, unsafe_allow_html=True)

if used_hours > 24.0:
    st.error("⚠️ Time allocation exceeds 24 hours! Please adjust your schedule.")
elif abs(used_hours - 24.0) < 0.1:
    st.success("🎯 Perfect! You've allocated your full day.")
else:
    st.info(f"⏳ You have {remaining_hours:.1f} hours of unallocated time.")

# Prediction section
st.markdown("---")

if st.button("🚀 Predict My Exam Score", use_container_width=True):
    if used_hours > 24.0:
        st.error("Please adjust your time allocation to stay within 24 hours")
    else:
        with st.spinner("🤖 AI is analyzing your study pattern..."):
            # Add some dramatic effect
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            try:
                input_data = np.array([[
                    st.session_state.study_hours, 
                    exercise_frequency,
                    st.session_state.social_media_hours, 
                    st.session_state.netflix_hours,
                    st.session_state.sleep_hours, 
                    mental_health_rating, 
                    attendance_percentage
                ]])
                
                input_scaled = scaler.transform(input_data)
                raw_prediction = model.predict(input_scaled)[0]
                capped_score = min(100, max(0, raw_prediction))
                
                # Enhanced prediction display
                if capped_score >= 90:
                    emoji = "🌟"
                    grade = "Excellent"
                    color = "#10b981"
                elif capped_score >= 70:
                    emoji = "👍"
                    grade = "Good"
                    color = "#3b82f6"
                elif capped_score >= 50:
                    emoji = "📚"
                    grade = "Average"
                    color = "#f59e0b"
                else:
                    emoji = "💪"
                    grade = "Needs Improvement"
                    color = "#ef4444"
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="margin: 0; font-size: 1.5rem;">Predicted Exam Score {emoji}</h2>
                    <div class="score-display">{capped_score:.1f}/100</div>
                    <div style="background: rgba(255,255,255,0.2); border-radius: 15px; height: 10px; margin: 1rem 0; position: relative; z-index: 1;">
                        <div style="background: white; width: {capped_score}%; height: 100%; border-radius: 15px; transition: width 0.5s ease;"></div>
                    </div>
                    <h3 style="margin: 0; font-size: 1.3rem;">{grade} Performance</h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Based on your current lifestyle pattern</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.history.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "score": capped_score,
                    "study": st.session_state.study_hours,
                    "sleep": st.session_state.sleep_hours,
                    "social_media": st.session_state.social_media_hours,
                    "entertainment": st.session_state.netflix_hours,
                    "exercise": exercise_frequency,
                    "mental_health": mental_health_rating,
                    "attendance": attendance_percentage
                })
                
                st.balloons()
                
                # Performance insights
                if show_tips:
                    st.markdown('<div class="section-header">💡 Personalized Insights</div>', unsafe_allow_html=True)
                    
                    insight_cols = st.columns(2)
                    insights_shown = False
                    
                    with insight_cols[0]:
                        # Study insights
                        if st.session_state.study_hours < 3:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>📖 Study Time</strong><br>
                                Consider increasing to 3-6 hours daily for better retention
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        elif st.session_state.study_hours >= 6:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>📖 Study Time</strong><br>
                                Great job! You're dedicating good time to studying
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        
                        if st.session_state.sleep_hours < 6:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>😴 Sleep Quality</strong><br>
                                Aim for 7-9 hours - sleep is crucial for memory consolidation
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        elif st.session_state.sleep_hours >= 8:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>😴 Sleep Quality</strong><br>
                                Excellent! You're getting enough sleep for optimal learning
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        
                        if exercise_frequency < 2:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>🏃 Physical Activity</strong><br>
                                Add 2-3 exercise sessions weekly to boost cognitive function
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        elif exercise_frequency >= 4:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>🏃 Physical Activity</strong><br>
                                Great! Regular exercise is boosting your cognitive performance
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                    
                    with insight_cols[1]:
                        if st.session_state.social_media_hours > 3:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>📱 Digital Wellness</strong><br>
                                Consider reducing social media to improve focus during study
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        elif st.session_state.social_media_hours <= 1:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>📱 Digital Wellness</strong><br>
                                Excellent digital discipline! This helps maintain focus
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        
                        if mental_health_rating < 5:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>🧘 Mental Health</strong><br>
                                Consider stress management techniques or seeking support
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        elif mental_health_rating >= 8:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>🧘 Mental Health</strong><br>
                                Great mental wellbeing! This positively impacts your learning
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        
                        if attendance_percentage < 80:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>🏫 Class Attendance</strong><br>
                                Improve attendance - it's strongly linked to better scores
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                        elif attendance_percentage >= 95:
                            st.markdown("""
                            <div class="tip-card">
                                <strong>🏫 Class Attendance</strong><br>
                                Outstanding attendance! You're maximizing learning opportunities
                            </div>
                            """, unsafe_allow_html=True)
                            insights_shown = True
                    
                    # Show general insights if no specific ones were triggered
                    if not insights_shown:
                        st.markdown("""
                        <div class="tip-card">
                            <strong>🌟 Overall Assessment</strong><br>
                            Your lifestyle balance looks good! Keep maintaining these healthy habits for consistent academic performance.
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please check your input values and try again")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; padding: 1rem;">
    <p>🎯 Made with ❤️ using Streamlit and AI • Your data stays private and secure</p>
</div>
""", unsafe_allow_html=True)