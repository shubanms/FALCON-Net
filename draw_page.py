import streamlit as st

def draw_character_attack_page():
    import numpy as np
    from PIL import Image
    from streamlit_drawable_canvas import st_canvas
    from attacks import apply_attack
    from image_utils import compute_mse, compute_difference_heatmap

    st.title("🖌️ Draw Character & Attack Playground")

    st.markdown("""
    <style>
    .draw-info {
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
    <div class="draw-info">
        <b>Interactive Drawing Workspace:</b><br>
        1. <b>Canvas</b>: Draw any character using the freehand drawing tool.<br>
        2. <b>Attack Selection</b>: Choose between FGSM and PGD attacks.<br>
        3. <b>Strength Control</b>: Adjust the intensity of the attack.<br>
        4. <b>Analysis</b>: Visualize the attack's impact through heatmaps and metrics.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "> **How to use this page:**\n"
        "> - Use your mouse or touch input to draw a character.\n"
        "> - Select an attack type and adjust its strength.\n"
        "> - Apply the attack to see how it affects your drawing.\n"
        "> - Analyze the differences using the pixel-level analysis tools.\n"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Draw Your Character")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        st.subheader("Your Drawing")
        if canvas_result.image_data is not None:
            drawn_img = Image.fromarray(
                (canvas_result.image_data[:, :, 0]).astype("uint8")
            ).convert("RGB")
            st.image(drawn_img, caption="Final Character", use_container_width=False)

    st.markdown("---")
    st.header("⚔️ Attack Configuration")

    attack_type = st.selectbox(
        "Choose Attack Type", ["FGSM", "PGD"]
    )
    attack_strength = st.slider("Attack Strength", 0.0, 10.0, 5.0)

    if canvas_result.image_data is not None and st.button("Apply Attack"):
        attacked_img = apply_attack(drawn_img, attack_type, attack_strength)

        st.markdown("### Original vs Attacked Image")
        col3, col4 = st.columns([1, 1])
        with col3:
            st.image(drawn_img, caption="Original Image", use_container_width=True)
        with col4:
            st.image(attacked_img, caption="Attacked Image", use_container_width=True)

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

        mse = compute_mse(drawn_img, attacked_img)
        st.markdown(
            f"<h5 style='text-align: center;'>MSE: {mse:.2f}</h5>",
            unsafe_allow_html=True,
        )

        fig = compute_difference_heatmap(drawn_img, attacked_img, small=True)
        st.pyplot(fig)
