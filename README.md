# 🦅 FALCON-Net

Welcome to FALCON-Net! 🚀

An interactive Streamlit playground for visualizing and testing the robustness of Siamese and Prototypical neural networks under adversarial attacks. 

🌐 **Live Demo:** [FALCONNet](https://falcon-net.streamlit.app)

---

## 🎯 What is FALCON-Net?

FALCON-Net is your go-to app for exploring how cool neural networks (Siamese & Prototypical) handle sneaky adversarial attacks! Draw, attack, visualize, and learn—all in one place. Perfect for students, researchers, and the just-plain-curious. 😎

---

## ✨ Features

- 🎨 **Draw & Attack:** Doodle your own character, unleash attacks (FGSM, PGD), and see the chaos unfold with heatmaps and metrics!
- 🗂️ **Pick & Attack:** Choose from a gallery of multilingual characters, attack them, and compare before/after results.
- 📊 **Metrics Dashboard:** Dive into interactive charts showing accuracy, robustness, and more.
- 🧬 **Siamese Network Explorer:** Peek inside each layer, visualize feature maps, and compare embeddings.
- 🧑‍🤝‍🧑 **Prototypical Network Explorer:** Play with few-shot learning, support/query sets, and see how prototypes work.

---

## 🛠️ Tech Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/) 🎈
- **Deep Learning:** [TensorFlow/Keras](https://keras.io/) 🧠
- **Visualization:** [Matplotlib](https://matplotlib.org/), [Pillow](https://python-pillow.org/) 📷
- **Data:** Sample character images (assets/) 🖼️

---

## 🚦 Get Started

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/FALCON-Net.git
cd FALCON-Net
```

### 2. Install the Magic
We recommend a virtual environment! 🪄
```bash
pip install -r requirements.txt
```

### 3. Launch the App
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` and let the fun begin! 🎉

---

## 🗂️ App Map

- `app.py` — Main hub, navigation, and page routing
- `draw_page.py` — Freehand drawing & attack fun
- `select_page.py` — Pick a character & attack
- `metrics_page.py` — Performance dashboards
- `siamese_page.py` — Siamese network explorer
- `prototypical_page.py` — Prototypical network explorer
- `attacks.py` — Adversarial attack code (FGSM, PGD)
- `image_utils.py` — Image helpers
- `assets/` — Character images

---

## 🕹️ How to Play

- Use the sidebar to jump between pages 🧭
- Follow the on-screen prompts to draw, attack, and explore
- Try different attacks and see how the models react
- Have fun and learn something new! 🤓

---

## 🚀 Deployment

The app is live on Streamlit Cloud:
[FALCONNet](https://falcon-net.streamlit.app)

Want your own version? Fork this repo and connect to [Streamlit Community Cloud](https://streamlit.io/cloud) in minutes!

---

## 📜 License

Licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Inspired by the amazing world of adversarial robustness & few-shot learning
- Built with Streamlit, TensorFlow/Keras, and open-source magic ✨

---

