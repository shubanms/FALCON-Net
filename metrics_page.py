import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def metrics_visualization_page():
    st.title("📊 Metrics & Visualizations")

    st.markdown("""
    <style>
    .metrics-info {
        background: #f7fafd;
        border-radius: 10px;
        border: 1px solid #e3e8ee;
        padding: 18px 10px 10px 10px;
        margin-bottom: 18px;
        font-family: 'Fira Mono', 'Consolas', monospace;
        font-size: 15px;
        color: #222;
        overflow-x: auto;
    }
    </style>
    <div class="metrics-info">
        <b>Performance Analysis Dashboard:</b><br>
        1. <b>Model Comparison</b>: Compare Siamese vs Prototypical Networks.<br>
        2. <b>Attack Resistance</b>: Analyze model performance under different attacks.<br>
        3. <b>Interactive Filtering</b>: Filter results by network type and evaluation mode.<br>
        4. <b>Visual Analytics</b>: Multiple visualizations of model performance metrics.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "> **How to use this dashboard:**\n"
        "> - Use the filters to focus on specific networks or evaluation types.\n"
        "> - Compare performance metrics across different model variants.\n"
        "> - Analyze the impact of different attacks on model accuracy.\n"
        "> - Explore detailed visualizations of model behavior.\n"
    )

    # Performance Metrics Section
    st.header("💫 Performance Comparison")
    st.markdown("""
        This section presents a detailed comparison between Siamese and Prototypical Networks,
        evaluating their performance under different conditions:
        - Base model performance
        - Adversarial Training (AT) impact
        - Data Diversification (DD) effects
        - Combined AT+DD effectiveness
    """)

    # Metrics Table
    st.header("Performance Comparison of Siamese and Prototypical Networks")

    data = {
        "Network": ["Siamese"] * 8 + ["Prototypical"] * 8,
        "Eval. Type": ["Test"] * 4 + ["Train"] * 4 + ["Test"] * 4 + ["Train"] * 4,
        "Model": [
            "Base Model", "AT Model", "DD Model", "AT + DD Model",
            "Base Model", "AT Model", "DD Model", "AT + DD Model",
            "Base Model", "AT Model", "DD Model", "AT + DD Model",
            "Base Model", "AT Model", "DD Model", "AT + DD Model",
        ],
        "No Attack Accuracy": [
            0.82, 0.79, 0.80, 0.77, 0.84, 0.82, 0.83, 0.81,
            0.96, 0.93, 0.94, 0.92, 0.98, 0.97, 0.98, 0.97
        ],
        "PGD Attack Accuracy": [
            0.50, 0.63, 0.48, 0.68, 0.53, 0.74, 0.55, 0.73,
            0.39, 0.64, 0.32, 0.73, 0.96, 0.83, 0.62, 0.83
        ],
        "FGSM Attack Accuracy": [
            0.50, 0.68, 0.58, 0.69, 0.56, 0.74, 0.63, 0.75,
            0.45, 0.72, 0.57, 0.71, 0.97, 0.65, 0.74, 0.89
        ]
    }

    df = pd.DataFrame(data)

    # Add filters in a single row above the table
    st.markdown("---")
    st.header("Filter Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        network_filter = st.multiselect("Select Network", options=df["Network"].unique(), default=df["Network"].unique())
    with col2:
        eval_type_filter = st.multiselect("Select Evaluation Type", options=df["Eval. Type"].unique(), default=df["Eval. Type"].unique())
    with col3:
        model_filter = st.multiselect("Select Model", options=df["Model"].unique(), default=df["Model"].unique())

    filtered_df = df[
        (df["Network"].isin(network_filter)) &
        (df["Eval. Type"].isin(eval_type_filter)) &
        (df["Model"].isin(model_filter))
    ]

    st.dataframe(filtered_df)

    # Visualizations
    st.header("Visualizations")

    # Two-column layout for the first two graphs
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Average No Attack Accuracy by Network")
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller size
        df.groupby("Network")["No Attack Accuracy"].mean().plot(kind="bar", ax=ax, color=["blue", "orange"])
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

    with col5:
        st.subheader("Impact of Attacks on Accuracy")
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller size
        for model in df["Model"].unique():
            subset = df[df["Model"] == model]
            ax.plot(["No Attack", "PGD Attack", "FGSM Attack"], subset.iloc[0, 3:6], label=model)
        ax.set_ylabel("Accuracy")
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Adjusted y-axis for 0 to 1 range
        ax.legend(fontsize="small", loc="upper right")  # Smaller legend
        st.pyplot(fig)

    # Additional graphs for PGD and FGSM attack accuracy
    col6, col7 = st.columns(2)
    with col6:
        st.subheader("Average PGD Attack Accuracy by Network")
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller size
        df.groupby("Network")["PGD Attack Accuracy"].mean().plot(kind="bar", ax=ax, color=["green", "red"])
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

    with col7:
        st.subheader("Average FGSM Attack Accuracy by Network")
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller size
        df.groupby("Network")["FGSM Attack Accuracy"].mean().plot(kind="bar", ax=ax, color=["purple", "cyan"])
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)