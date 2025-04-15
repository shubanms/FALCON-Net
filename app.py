"""
Falconnet Demo - Interactive Neural Network Visualization & Attack Analysis
-----------------------------------------------------------------------
An interactive application for visualizing and analyzing the behavior of Siamese and 
Prototypical Networks under various adversarial attacks.

Key Features:
1. Draw Character & Attack: Interactive drawing interface with attack visualization
2. Choose Character & Attack: Pre-defined character selection and attack analysis
3. Metrics & Visualizations: Comprehensive performance analysis dashboard
4. Siamese Network Visualization: Deep dive into Siamese network behavior
5. Prototypical Network Visualization: Analysis of Prototypical network functioning
"""

import streamlit as st
from draw_page import draw_character_attack_page
from select_page import select_character_attack_page
from metrics_page import metrics_visualization_page
from siamese_page import siamese_network_page
from prototypical_page import prototypical_network_page

# Configure the main page layout
st.set_page_config(
    layout="wide",
    page_title="Falconnet Demo",
    page_icon="🦅",
)

def home_page():
    st.title("🦅 Welcome to Falconnet Demo")
    
    # Hero section with description
    st.markdown("""
    <style>
    .hero {
        background: #f7fafd;
        border-radius: 10px;
        border: 1px solid #e3e8ee;
        padding: 25px;
        margin-bottom: 30px;
    }
    </style>
    <div class="hero">
        <h2>About Falconnet</h2>
        <p style="font-size: 1.2em;">
        This demo showcases the behavior and robustness of different neural network architectures 
        under adversarial attacks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("🚀 Explore Different Aspects of Neural Networks")
    st.markdown("Choose a section below to start exploring:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎨 Draw Character & Attack", use_container_width=True):
            st.session_state.page = "Draw Character & Attack"
        if st.button("📊 Metrics & Visualizations", use_container_width=True):
            st.session_state.page = "Metrics & Visualizations"
        if st.button("🌐 Prototypical Network Visualization", use_container_width=True):
            st.session_state.page = "Prototypical Network Visualization"
    
    with col2:
        if st.button("📂 Choose Character & Attack", use_container_width=True):
            st.session_state.page = "Choose Character & Attack"
        if st.button("🔗 Siamese Network Visualization", use_container_width=True):
            st.session_state.page = "Siamese Network Visualization"

    # Add some descriptive content below
    st.markdown("---")
    st.header("✨ Key Features")
    
    features = {
        "Draw Character & Attack": "Create your own characters and test network resilience against attacks",
        "Choose Character & Attack": "Experiment with pre-defined characters and various attack types",
        "Metrics & Visualizations": "Analyze comprehensive performance metrics and visual comparisons",
        "Siamese Network Visualization": "Deep dive into Siamese network architecture and behavior",
        "Prototypical Network Visualization": "Explore how prototypical networks learn and classify"
    }
    
    for title, desc in features.items():
        st.markdown(f"**{title}**  \n{desc}")

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Add a home button in the sidebar
if st.sidebar.button('🏠 Home'):
    st.session_state.page = 'Home'

# Route to appropriate page
if st.session_state.page == 'Home':
    home_page()
elif st.session_state.page == "Draw Character & Attack":
    draw_character_attack_page()
elif st.session_state.page == "Choose Character & Attack":
    select_character_attack_page()
elif st.session_state.page == "Metrics & Visualizations":
    metrics_visualization_page()
elif st.session_state.page == "Siamese Network Visualization":
    siamese_network_page()
elif st.session_state.page == "Prototypical Network Visualization":
    prototypical_network_page()
