import streamlit as st


def select_character_attack_page():
    from PIL import Image
    from attacks import apply_attack
    from image_utils import compute_mse, compute_difference_heatmap, load_sample_characters

    st.title("📂 Choose Character & Attack Playground")

    st.markdown("""
    <style>
    .select-info {
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
    <div class="select-info">
        <b>Character Selection & Attack Analysis:</b><br>
        1. <b>Character Selection</b>: Choose from a diverse set of characters.<br>
        2. <b>Attack Configuration</b>: Apply FGSM or PGD attacks.<br>
        3. <b>Visual Comparison</b>: Compare original vs attacked images.<br>
        4. <b>Impact Analysis</b>: Analyze attack effects through metrics and heatmaps.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "> **Using this playground:**\n"
        "> - Select a character from the available options.\n"
        "> - Configure and apply adversarial attacks.\n"
        "> - Compare the original and attacked versions.\n"
        "> - Analyze the attack impact using visual tools.\n"
    )

    characters = load_sample_characters()
    char_names = list(characters.keys())

    char_classes = {
        "Character 1": "Arcadian Character",
        "Character 2": "Bengali Character",
        "Character 3": "Braille Character",
        "Character 4": "Greek Character",
        "Character 5": "Japanese Character",
    }

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Select a Character")
        selected_char = st.selectbox("Choose one", char_names)
        selected_image = characters[selected_char]

    with col2:
        st.subheader("Selected Character")
        st.image(selected_image, caption="Original Image", use_container_width=True)
        st.markdown(f"<h5 style='text-align: center; color: #555;'>Class: {char_classes[selected_char]}</h5>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("⚔️ Attack Configuration")

    attack_type = st.selectbox(
        "Choose Attack Type", ["FGSM", "PGD"]
    )
    attack_strength = st.slider("Attack Strength", 0.0, 10.0, 5.0)  # Adjusted range to 0-10

    if st.button("Apply Attack"):
        attacked_image = apply_attack(selected_image, attack_type, attack_strength)

        st.markdown("### Original vs Attacked Image")
        col3, col4 = st.columns([1, 1])
        with col3:
            st.image(selected_image, caption="Original", use_container_width=True)
        with col4:
            st.image(attacked_image, caption="Attacked", use_container_width=True)

        st.markdown("---")
        st.header("📊 Pixel-Level Analysis")

        st.markdown(
            """<style>
            .pixel-analysis {
                font-size: 1.1em;
                color: #333;
                background-color: #f9f9f9;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            </style>""",
            unsafe_allow_html=True
        )

        st.markdown(
            """<div class="pixel-analysis">
            <b>Pixel-Level Analysis</b> provides insights into how the attack has altered the image.<br>
            - <b>MSE (Mean Squared Error):</b> Measures the average squared difference between the original and attacked images.<br>
            - <b>Heatmap:</b> Highlights the regions most affected by the attack.
            </div>""",
            unsafe_allow_html=True
        )

        mse = compute_mse(selected_image, attacked_image)
        st.markdown(
            f"<h5 style='text-align: center;'>MSE: {mse:.2f}</h5>",
            unsafe_allow_html=True,
        )

        fig = compute_difference_heatmap(selected_image, attacked_image, small=True)
        st.pyplot(fig)
