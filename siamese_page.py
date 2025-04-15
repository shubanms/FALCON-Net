import streamlit as st

def build_siamese_network():
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, LeakyReLU
    from keras import backend as K

    input_shape = (28, 28, 1)
    input_a = Input(shape=input_shape, name="Input_A")
    input_b = Input(shape=input_shape, name="Input_B")

    # Create shared layers
    conv1 = Conv2D(8, (3, 3), padding='same')
    leaky1 = LeakyReLU()
    pool1 = MaxPooling2D((2, 2))
    conv2 = Conv2D(16, (3, 3), padding='same')
    leaky2 = LeakyReLU()
    pool2 = MaxPooling2D((2, 2))
    flatten = Flatten()
    dense = Dense(8, activation='relu')

    def shared_network(input_layer):
        x = conv1(input_layer)
        x = leaky1(x)
        x = pool1(x)
        x = conv2(x)
        x = leaky2(x)
        x = pool2(x)
        x = flatten(x)
        x = dense(x)
        return x

    # Process both inputs
    processed_a = shared_network(input_a)
    processed_b = shared_network(input_b)

    l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
    output = Dense(1, activation='sigmoid')(l1_distance)

    model = Model(inputs=[input_a, input_b], outputs=output)
    return model

def build_visualization_network():
    from keras.models import Model
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU

    input_shape = (28, 28, 1)
    input_layer = Input(shape=input_shape, name="vis_input")
    
    # Create visualization layers with unique names
    x = Conv2D(8, (3, 3), padding='same', name='vis_conv1')(input_layer)
    x = LeakyReLU(name='vis_leaky1')(x)
    x = MaxPooling2D((2, 2), name='vis_pool1')(x)
    x = Conv2D(16, (3, 3), padding='same', name='vis_conv2')(x)
    x = LeakyReLU(name='vis_leaky2')(x)
    x = MaxPooling2D((2, 2), name='vis_pool2')(x)
    x = Flatten(name='vis_flatten')(x)
    x = Dense(8, activation='relu', name='vis_dense')(x)
    
    model = Model(inputs=input_layer, outputs=x, name='visualization_model')
    return model

def siamese_network_page():
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from image_utils import load_sample_characters
    from attacks import apply_attack
    from keras.models import Model

    st.title("🔗 Siamese Network Visualization")

    st.markdown("""
    <style>
    .siamese-arch {
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
    <div class="siamese-arch">
        <b>Siamese Network Architecture:</b><br>
        1. <b>Input Images</b>: Two images are processed in parallel.<br>
        2. <b>Shared Network</b>: Both images go through identical CNN layers.<br>
        3. <b>Feature Extraction</b>: Each image is converted to a feature vector.<br>
        4. <b>Distance Computation</b>: Calculate similarity between feature vectors.<br>
        5. <b>Layer Visualization</b>: Inspect network's internal representations.<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "> **How to use this page:**\n"
        "> - Select two images for comparison.\n"
        "> - Optionally apply attacks to either image.\n"
        "> - Explore how the network processes each image through different layers.\n"
        "> - Analyze feature maps and embeddings at each stage.\n"
    )

    # Load sample characters
    characters = load_sample_characters()
    char_names = list(characters.keys())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select First Image")
        selected_char_a = st.selectbox("Choose Image A", char_names, key="image_a")
        image_a = characters[selected_char_a]
        # Attack options for Image A
        attack_type_a = st.selectbox("Attack Type for Image A", ["None", "FGSM", "PGD"], key="attack_a")
        attack_strength_a = st.slider("Attack Intensity for Image A", 0.0, 10.0, 0.0, key="strength_a")
        attacked_image_a = apply_attack(image_a, attack_type_a, attack_strength_a) if attack_type_a != "None" and attack_strength_a > 0 else image_a
        st.image(attacked_image_a, caption="Image A (Attacked)", use_container_width=True)

    with col2:
        st.subheader("Select Second Image")
        selected_char_b = st.selectbox("Choose Image B", char_names, key="image_b")
        image_b = characters[selected_char_b]
        # Attack options for Image B
        attack_type_b = st.selectbox("Attack Type for Image B", ["None", "FGSM", "PGD"], key="attack_b")
        attack_strength_b = st.slider("Attack Intensity for Image B", 0.0, 10.0, 0.0, key="strength_b")
        attacked_image_b = apply_attack(image_b, attack_type_b, attack_strength_b) if attack_type_b != "None" and attack_strength_b > 0 else image_b
        st.image(attacked_image_b, caption="Image B (Attacked)", use_container_width=True)

    st.markdown("---")
    st.header("Layer-by-Layer Visualization")
    st.markdown(
        "> **Inspect the outputs for each image at each major phase.**\n"
        "> - Select a phase to view the intermediate output for both images.\n"
    )

    # Preprocess images
    def preprocess_image(image):
        img_array = np.array(image.convert("L").resize((28, 28))).astype("float32") / 255.0
        return np.expand_dims(img_array, axis=(0, -1))

    img_a = preprocess_image(attacked_image_a)
    img_b = preprocess_image(attacked_image_b)

    # Build the visualization network
    vis_model = build_visualization_network()

    # Define the visualization phases
    visualization_phases = [
        ("vis_conv1", "After First Conv", "Feature maps after the first convolution"),
        ("vis_leaky1", "After First LeakyReLU", "After first activation"),
        ("vis_pool1", "After First MaxPool", "After first pooling"),
        ("vis_conv2", "After Second Conv", "Feature maps after the second convolution"),
        ("vis_leaky2", "After Second LeakyReLU", "After second activation"),
        ("vis_pool2", "After Second MaxPool", "After second pooling"),
        ("vis_flatten", "After Flatten", "Flattened features"),
        ("vis_dense", "After Dense(8)", "Final feature embedding")
    ]

    # UI for phase selection
    phase_labels = [label for _, label, _ in visualization_phases]
    phase = st.selectbox("Select Phase to Visualize", phase_labels)
    phase_idx = phase_labels.index(phase)
    layer_name, _, layer_info = visualization_phases[phase_idx]

    st.info(layer_info)

    # Create intermediate model for the selected phase
    intermediate_model = Model(
        inputs=vis_model.input,
        outputs=vis_model.get_layer(layer_name).output,
        name=f'intermediate_{layer_name}'
    )

    # Get outputs for both images
    output_a = intermediate_model.predict(img_a)
    output_b = intermediate_model.predict(img_b)

    colA, colB = st.columns(2)
    
    def display_output(output, title, column):
        with column:
            st.markdown(f"**{title}**")
            if len(output.shape) == 4:
                # For convolutional and pooling layers - show feature maps as images
                num_filters = output.shape[-1]
                st.write(f"{num_filters} Feature Maps (showing up to 5)")
                n_display = min(num_filters, 5)
                fig, axes = plt.subplots(1, n_display, figsize=(15, 5))
                if n_display == 1:
                    axes = [axes]
                for i in range(n_display):
                    axes[i].imshow(output[0, :, :, i], cmap="viridis")
                    axes[i].axis("off")
                st.pyplot(fig)
            elif len(output.shape) == 2:
                # For flattened and dense layers - show feature distributions
                st.write(f"Feature Vector Shape: {output.shape}")
                if layer_name == 'vis_flatten':
                    st.markdown("""
                        > **Flattened Features:**  
                        > This histogram shows the distribution of values after converting the 2D feature maps into a 1D vector.
                        > Each bar represents one flattened pixel value from the previous layer's feature maps.
                    """)
                elif layer_name == 'vis_dense':
                    st.markdown("""
                        > **Dense Layer Features:**  
                        > This histogram shows the learned feature values after dimensionality reduction.
                        > Each bar represents a different learned high-level feature that the network uses for comparison.
                    """)
                
                # Create a more informative histogram
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(output[0], bins=30, edgecolor='black')
                ax.set_title(f'Distribution of {layer_name} Features')
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Count')
                st.pyplot(fig)
                
                # Also show the actual feature values
                st.write("Feature Values:")
                st.line_chart(output[0])
            else:
                st.write(f"Output: {output}")

    display_output(output_a, "Image A", colA)
    display_output(output_b, "Image B", colB)

    st.markdown("---")

