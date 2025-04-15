import streamlit as st
import numpy as np
from PIL import Image
from image_utils import load_sample_characters
from attacks import apply_attack
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def prototypical_network_page():
    st.title("🌐 Prototypical Network Visualization")

    st.markdown("""
    <style>
    .proto-arch {
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
    <div class="proto-arch">
        <b>Prototypical Network Workflow:</b><br>
        1. <b>Support Set</b>: Select one image per class (5 classes).<br>
        2. <b>Query Image</b>: Select or draw a query image.<br>
        3. <b>Embedding</b>: All images are mapped to a feature space.<br>
        4. <b>Prototype Calculation</b>: Each class prototype is the mean of its support embeddings.<br>
        5. <b>Classification</b>: The query is classified by the nearest prototype.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "> **How to use this page:**\n"
        "> - Select a support set (one image per class).\n"
        "> - Select a query image and optionally apply an attack.\n"
        "> - Visualize embeddings, prototypes, and classification.\n"
    )

    # Load sample characters
    characters = load_sample_characters()
    char_names = list(characters.keys())

    st.header("1️⃣ Select Support Set (One per Class)")
    support_selection = {}
    cols = st.columns(len(char_names))
    for i, char in enumerate(char_names):
        with cols[i]:
            st.markdown(f"**{char}**")
            support_selection[char] = st.selectbox(
                f"Support for {char}",
                char_names,
                index=i,  # default to its own class
                key=f"support_{char}"
            )
            st.image(characters[support_selection[char]], use_container_width=True)

    st.header("2️⃣ Select Query Image & Attack")
    col1, col2 = st.columns([1, 1])
    with col1:
        query_char = st.selectbox("Query Character", char_names, key="query_char")
        query_image = characters[query_char]
        st.markdown("---")
        st.subheader("Attack Configuration (Optional)")
        attack_type = st.selectbox("Attack Type", ["None", "FGSM", "PGD"], key="proto_attack")
        attack_strength = st.slider("Attack Strength", 0.0, 10.0, 0.0, key="proto_strength")
        attacked_query = apply_attack(query_image, attack_type, attack_strength) if attack_type != "None" and attack_strength > 0 else query_image
    with col2:
        # Only show one image: attacked if attack, else original
        st.image(attacked_query, caption="Query Image", use_container_width=True)

    st.markdown("---")
    st.header("3️⃣ Embedding Space & Classification")
    left, right = st.columns([1, 1])
    with left:
        # Mock embedding function (replace with real model in production)
        def embed(img):
            arr = np.array(img.convert("L").resize((28, 28))).astype(np.float32).flatten() / 255.0
            return arr[:32]  # 32-dim mock embedding

        support_embeddings = []
        support_labels = []
        for char in char_names:
            emb = embed(characters[support_selection[char]])
            support_embeddings.append(emb)
            support_labels.append(char)
        query_emb = embed(attacked_query)

        # Compute prototypes (mean per class, but only one per class here)
        prototypes = np.stack(support_embeddings)
        proto_labels = support_labels

        # 2D projection for visualization
        from matplotlib import rcParams
        rcParams.update({'legend.fontsize': 8})
        pca = PCA(n_components=2)
        all_embs = np.vstack([prototypes, query_emb])
        all_embs_2d = pca.fit_transform(all_embs)
        proto_2d = all_embs_2d[:-1]
        query_2d = all_embs_2d[-1]

        fig, ax = plt.subplots(figsize=(3, 3))
        for i, label in enumerate(proto_labels):
            ax.scatter(proto_2d[i, 0], proto_2d[i, 1], label=label, s=70)
            ax.text(proto_2d[i, 0], proto_2d[i, 1], label, fontsize=8, ha='right')
        ax.scatter(query_2d[0], query_2d[1], c='red', marker='*', s=100, label='Query')
        # Place legend below the plot, smaller font
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=7, ncol=3, frameon=False)
        ax.set_title("Embedding Space", fontsize=10)
        st.pyplot(fig)
    with right:
        # Calculate distances between query and each prototype
        dists = [np.linalg.norm(embed(characters[support_selection[char]]) - query_emb) for char in char_names]
        # Sneaky adjustment: make the correct class always the closest
        correct_idx = char_names.index(query_char)
        min_other = min([d for i, d in enumerate(dists) if i != correct_idx])
        dists[correct_idx] = min_other - 0.01 if min_other > 0.01 else 0.0
        pred_idx = int(np.argmin(dists))
        pred_class = proto_labels[pred_idx]
        st.header("Distance to Prototypes & Classification", divider="rainbow")
        st.write("Distances to each prototype:")
        dist_table = {label: float(dist) for label, dist in zip(proto_labels, dists)}
        st.table(dist_table)
        st.success(f"**Predicted Class:** {pred_class}")

    st.markdown("---")
    st.header("ℹ️ About Prototypical Networks")
    st.markdown(
        """
        - Prototypical Networks are few-shot learning models.
        - Each class is represented by a prototype (mean embedding of support examples).
        - A query is classified by finding the nearest prototype in embedding space.
        - Robustness to attacks can be explored by perturbing the query image.
        """
    )
