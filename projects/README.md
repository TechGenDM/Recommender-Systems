# 🚀 Recommender Systems: Capstone & Portfolio Projects

Welcome to the **Projects Hub** of the [Recommender Systems Curriculum](../README.md)!

While theoretical understanding of matrix factorization and deep architectures is essential, **building, benchmarking, and deploying recommender models on real-world datasets** is what separates learners from production engineers.

This directory contains a progressive sequence of 5 industry-standard capstone projects spanning classical collaborative filtering, implicit feedback modeling, large-scale two-tower deep retrieval, latent space visualization, and ethical guardrails.

---

## 🗺️ Project Roadmap at a Glance

| # | Project Title | Architecture / Paradigm | Dataset | Key Deliverable | Difficulty | Status |
|:---:|:---|:---|:---|:---|:---:|:---:|
| **01** | [**CineMatch**](#project-1-cinematch--vectorized-collaborative-filtering-from-scratch) | Matrix Factorization & Mean Normalization | MovieLens 100K | Custom AutoDiff Engine & Related Items Finder | 🟡 Intermediate | ⏳ Next Up |
| **02** | [**ClickStream Rec**](#project-2-clickstream-rec--implicit-feedback-recommender) | Implicit Feedback & Binary Cross-Entropy | Retailrocket / Instacart | Negative Sampling & Next-Item Predictor | 🟡 Intermediate | 📝 Planned |
| **03** | [**StreamPulse**](#project-3-streampulse--two-tower-deep-retrieval-with-vector-search) | Two-Tower Deep Neural Network + ANN | MovieLens 1M / Spotify | Sub-10ms Candidate Retrieval (ScaNN / FAISS) | 🔴 Advanced | 📝 Planned |
| **04** | [**EmbeddingScope**](#project-4-embeddingscope--interactive-3d-latent-space-explorer) | PCA & Dimensionality Reduction | MovieLens Genome | Interactive 3D Latent Space Visualizer | 🟢 Foundational | 📝 Planned |
| **05** | [**FairRec**](#project-5-fairrec--algorithmic-bias-audit--serendipity-guardrails) | Ethical Guardrails & Re-ranking | Synthetic + MovieLens | De-biasing & Filter Bubble Mitigation Suite | 🔴 Advanced | 📝 Planned |

---

## 📌 Detailed Project Specifications

### Project 1: CineMatch — Vectorized Collaborative Filtering from Scratch

> **Focus**: Mathematical foundations, joint optimization, and handling cold users with mean normalization.

* **🎯 Problem Statement**: Build a movie recommendation engine that learns latent representations of users and movies simultaneously without relying on pre-packaged recommendation libraries.
* **📊 Dataset**: [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/) (GroupLens Research)
  * 100,000 ratings (1–5 stars) from 943 users on 1,682 movies.
  * User-item interaction matrix sparsity $\approx 93.7\%$.
* **🛠️ Tech Stack**: Python, NumPy, Pandas, TensorFlow (`tf.GradientTape`, `tf.Variable`), Matplotlib.
* **🧠 Core Techniques**:
  * Matrix centering via **Item Mean Normalization** ($\mu_i$).
  * Vectorized cost function computation with binary masking matrix $R$.
  * Gradient descent optimization via Adam optimizer.
  * Nearest-neighbor item discovery using squared Euclidean distance:
    $$\|x^{(i)} - x^{(k)}\|^2$$
* **💡 Why This Project is Critical**:
  Most beginners use black-box libraries like `Surprise` or `scikit-learn`. Implementing joint matrix factorization and AutoDiff from first principles demonstrates deep algorithmic mastery to interviewers and recruiters.
* **📁 Directory**: `projects/01_cinematch_collaborative_filtering/`

---

### Project 2: ClickStream Rec — Implicit Feedback Recommender

> **Focus**: Real-world e-commerce, binary engagement signals (clicks, likes, adds-to-cart), and negative sampling.

* **🎯 Problem Statement**: In production platforms, explicit 5-star ratings are extremely rare. Learn user purchase intent strictly from binary interaction streams (clicks, views, and favorites).
* **📊 Dataset**: [Retailrocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) or [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)
  * Timestamped clickstream events: `view`, `addtocart`, `transaction`.
* **🛠️ Tech Stack**: Python, TensorFlow / Keras, SciPy Sparse, Scikit-learn.
* **🧠 Core Techniques**:
  * Binary conversion: $y \in \{0, 1\}$.
  * **Negative Sampling**: Synthesizing unobserved pairs to combat extreme positive-class bias.
  * Logistic Loss / Binary Cross-Entropy:
    $$J = -\sum_{(i,j)} \left[ y^{(i,j)} \log(\hat{y}^{(i,j)}) + (1 - y^{(i,j)}) \log(1 - \hat{y}^{(i,j)}) \right]$$
  * Top-$K$ Ranking Metrics: **HitRate@K**, **Precision@K**, and **Mean Reciprocal Rank (MRR)**.
* **💡 Why This Project is Critical**:
  Over 95% of real-world recommendation problems (Amazon, TikTok, Taobao) are implicit feedback problems. Demonstrating familiarity with implicit conversion and ranking metrics directly matches industry job descriptions.
* **📁 Directory**: `projects/02_clickstream_implicit_rec/`

---

### Project 3: StreamPulse — Two-Tower Deep Retrieval with Vector Search

> **Focus**: Modern industrial architecture (YouTube / Netflix candidate generation), deep user/item representations, and sub-10ms Approximate Nearest Neighbors (ANN).

* **🎯 Problem Statement**: How do you recommend relevant items from a catalogue of 1,000,000+ items to an active user in less than 20 milliseconds?
* **📊 Dataset**: [MovieLens 1M / 25M](https://grouplens.org/datasets/movielens/) or [Spotify Million Playlist Challenge Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).
* **🛠️ Tech Stack**: TensorFlow 2.x, Google ScaNN (or Facebook FAISS), FastAPI, Docker.
* **🧠 Core Techniques**:
  * **Two-Tower Architecture**:
    * **User Tower**: Deep dense network mapping user demographics + interaction history $\to v_u \in \mathbb{R}^{32}$.
    * **Item Tower**: Deep dense network mapping item metadata + genre embeddings $\to v_m \in \mathbb{R}^{32}$.
  * **L2 Vector Normalization**: Ensuring $\hat{y} = v_u \cdot v_m = \cos(v_u, v_m)$.
  * **Offline Precomputation**: Caching millions of item vectors into an indexed vector database.
  * **Approximate Nearest Neighbors (ANN)**: Real-time MIPS (Maximum Inner Product Search) with sub-10ms lookup.
* **💡 Why This Project is Critical**:
  The two-tower dual encoder is the gold-standard architecture used by Google, Meta, and Pinterest. This project demonstrates full-stack ML engineering: model design, vector indexing, and low-latency API serving.
* **📁 Directory**: `projects/03_twotower_deep_retrieval/`

---

### Project 4: EmbeddingScope — Interactive 3D Latent Space Explorer

> **Focus**: Unsupervised dimensionality reduction, PCA mathematical decomposition, and latent vector interpretability.

* **🎯 Problem Statement**: Deep recommender embeddings reside in 32-D to 128-D vector spaces, making them impossible for humans to interpret. Project them down to interactive 2D/3D spaces to audit learned genre clusters and semantic trajectories.
* **📊 Dataset**: Learned movie or music embeddings from Project 1 / Project 3.
* **🛠️ Tech Stack**: NumPy, Scikit-learn (`PCA`), Plotly, Streamlit.
* **🧠 Core Techniques**:
  * Zero-centering and feature standardization.
  * Covariance matrix computation: $\Sigma = \frac{1}{m} X^T X$.
  * Singular Value Decomposition (SVD) and Explained Variance Ratio analysis.
  * Interactive 3D scatter visualization color-coded by genre/tags.
* **💡 Why This Project is Critical**:
  Provides intuitive visual proof that latent factors encode genuine semantic meaning (e.g., separating "Romantic Comedies" from "Gritty Sci-Fi").
* **📁 Directory**: `projects/04_embeddingscope_pca/`

---

### Project 5: FairRec — Algorithmic Bias Audit & Serendipity Guardrails

> **Focus**: Responsible AI, filter bubble mitigation, exposure fairness, and diversification.

* **🎯 Problem Statement**: Unchecked optimization of click-through rate causes extreme popularity bias (superstars dominate the feed) and traps users in ideological/interest echo chambers.
* **📊 Dataset**: Synthetic user consumption logs + MovieLens.
* **🛠️ Tech Stack**: Python, NumPy, SciPy, Matplotlib, Seaborn.
* **🧠 Core Techniques**:
  * **Popularity Bias Measurement**: Gini coefficient of item exposure.
  * **Intra-List Diversity (ILD)**: Ensuring recommendations span multiple genres rather than monocultures.
  * **$\epsilon$-Greedy & Serendipity Injection**: Introducing calculated exploration to uncover latent user interests.
  * **Fair Re-Ranking Policy**: Calibrating candidate lists to ensure minority-category representation.
* **💡 Why This Project is Critical**:
  Ethical AI and bias mitigation are central concerns in modern platform design and regulatory compliance. Having this project showcases maturity beyond pure model fitting.
* **📁 Directory**: `projects/05_fairrec_ethical_guardrails/`

---

## 📐 Standardized Project Template

Every project repository will follow a consistent, production-ready structure:

```text
project_name/
├── README.md               <- Comprehensive documentation, math, results & benchmarks
├── data/
│   ├── raw/                <- Download scripts / instructions
│   └── processed/          <- Preprocessed matrices / TFRecord datasets
├── notebooks/
│   ├── 01_eda.ipynb        <- Exploratory Data Analysis & Sparsity Audit
│   └── 02_training.ipynb   <- Model development & evaluation walkthrough
├── src/
│   ├── model.py            <- TensorFlow / NumPy model definitions
│   ├── loss.py             <- Custom loss functions & metrics
│   ├── train.py            <- Training script with CLI flags
│   └── evaluate.py         <- Top-K ranking & similarity evaluation
└── requirements.txt        <- Pin-pointed dependencies
```

---

## 📚 Essential Reading & Benchmark Papers

To build world-class projects, these landmark research papers are highly recommended:

1. **Matrix Factorization Techniques for Recommender Systems**  
   *Yehuda Koren, Robert Bell, Chris Volinsky (Netflix Prize)* — [IEEE Computer, 2009](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
2. **Deep Neural Networks for YouTube Recommendations**  
   *Paul Covington, Jay Adams, Emre Sargin (Google)* — [ACM RecSys, 2016](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
3. **Collaborative Filtering for Implicit Feedback Datasets**  
   *Yifan Hu, Yehuda Koren, Chris Volinsky* — [IEEE ICDM, 2008](http://yifanhu.net/PUB/cf.pdf)
4. **Accelerating Large-Scale Inference with ScaNN (Score-aware Quantization)**  
   *Ruiqi Guo et al. (Google Research)* — [ICML, 2020](https://arxiv.org/abs/1908.10396)

---

## 🤝 Getting Involved

Want to build or collaborate on any of these implementations?
- Check the [Issues tab](https://github.com/TechGenDM/Recommender-Systems/issues) for starter tasks.
- Follow the [Curriculum Notes](../README.md) to understand each building block before running code.
