# 🎬 CineMatch Pro - AI Content Recommender

![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![TMDB](https://img.shields.io/badge/CDN-TMDB%20API-01B4E4.svg)

## 🚀 Project Overview
CineMatch Pro is a sophisticated content-based recommendation engine modeled after premium OTT platforms. By analyzing movie genres and plot descriptions using Natural Language Processing, the engine recommends the most mathematically relevant movies based on a user's search query.

*Developed as part of the AI & ML Internship at Elevate Labs.*

## 🧠 System Architecture & Methodology
1. **Metadata Aggregation:** Combines multiple contextual features into a single textual string.
2. **Vector Space Modeling:** Uses `TfidfVectorizer` to convert movie descriptions into a sparse matrix.
3. **Similarity Engine:** Applies Cosine Similarity to find the mathematical angle between vectors.

## 🔥 Key Features
* **Premium OTT UI:** Features a dark-mode, cinematic interface with functional hero trailers.
* **Match Scoring:** Dynamically calculates a real-time percentage match for recommendations.
* **High-Quality Rendering:** Integrates TMDB CDN links for flawless poster rendering.

## ⚙️ Installation & Usage
git clone https://github.com/your-username/CineMatch-Movie-Recommender.git
cd CineMatch-Movie-Recommender
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author
**Md Salman Farsi**
* **Role:** AI & ML Intern @ Elevate Labs | B.Tech CS (AI & ML)
* **Portfolio:** [mdsalmanfarsi.io](https://mdsalmanfarsi692004-svg.github.io/portfolio/)
* **Email:** [mdsalmanfarsi692004@gmail.com](mailto:mdsalmanfarsi692004@gmail.com)
* **GitHub:** [https://github.com/mdsalmanfarsi692004-svg]
* **LinkedIn:** [www.linkedin.com/in/md-salman-farsi-data-analyst]
