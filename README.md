# 🎯 Recommender Systems: From Mathematical Foundations to Deep Learning

[![GitHub Stars](https://img.shields.io/github/stars/TechGenDM/Recommender-Systems?style=for-the-badge&logo=github&color=blue)](https://github.com/TechGenDM/Recommender-Systems/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/TechGenDM/Recommender-Systems?style=for-the-badge&logo=github&color=indigo)](https://github.com/TechGenDM/Recommender-Systems/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Curriculum-success?style=for-the-badge)](#curriculum-roadmap)

> **A comprehensive, rigorous, and code-first open-access masterclass on Recommender Systems.**  
> Designed for college students, software engineers, and machine learning practitioners seeking to master both the mathematical intuition and production-grade implementation of recommendation engines.

---

## 🌟 Philosophy & Pedagogical Approach

Modern recommendation engines power the world's most critical digital experiences—from Netflix and YouTube to Spotify, Amazon, and TikTok. However, learning materials often swing between high-level conceptual hand-waving and opaque, black-box library tutorials.

This curriculum follows a battle-tested 4-pillar pedagogical framework:

1. **Intuition First**: Demystify *why* an algorithm works through geometric analogies and real-world scenarios before looking at code.
2. **Mathematical Rigor**: Derive cost functions, gradients, vector projections, and latent representations step-by-step.
3. **From-Scratch & TensorFlow Implementation**: Build algorithms end-to-end using standard numerical libraries and deep learning frameworks (`tf.GradientTape`, custom layers, and vector search).
4. **Systems & Ethics**: Tackle scale (large catalogues, latency constraints, retrieval vs. ranking) and responsibility (algorithmic bias, filter bubbles, feedback loops).

---

## 🏗️ High-Level System Architecture

```mermaid
flowchart TD
    subgraph DataInput ["1. User & Item Data"]
        U["User Profiles & History<br/>(Likes, Clicks, Ratings)"]
        I["Item Catalogue & Metadata<br/>(Genres, Text, Features)"]
    end

    subgraph RetrievalStage ["2. Candidate Generation (Retrieval)"]
        CF["Collaborative Filtering<br/>(Matrix Factorization / Latent Factors)"]
        TT["Two-Tower Deep Models<br/>(User & Item Vector Embeddings)"]
        ANN["Approximate Nearest Neighbors<br/>(Vector Search: ScaNN / FAISS)"]
        U --> CF
        I --> CF
        U --> TT
        I --> TT
        CF --> ANN
        TT --> ANN
    end

    subgraph RankingStage ["3. Scoring & Ranking"]
        Ranker["Content-Based & Deep Rankers<br/>(Feature Crosses, Dense Networks)"]
        ANN -->|"Top ~100 Candidates"| Ranker
    end

    subgraph ServingStage ["4. Re-Ranking & Serving"]
        Ethical["Diversity, Deduplication & Ethical Guardrails<br/>(Filter Bubble Mitigation, Fair Exposure)"]
        Recs["Final Recommendations Feed<br/>(Top-K Delivered to User)"]
        Ranker --> Ethical
        Ethical --> Recs
    end
```

---

## 📚 Complete Curriculum Roadmap

The modules below are ordered sequentially. Each topic builds directly on the theoretical and practical foundations laid by the previous chapters.

| # | Topic | Key Concepts | Focus | Status |
|:---:|:---|:---|:---:|:---:|
| **01** | [Making Recommendations](#01-making-recommendations) | Problem formulation, explicit vs. implicit feedback, recommendation matrix | 🧠 Concept | 📝 Planned |
| **02** | [Using Per-Item Features](#02-using-per-item-features) | Content-based linear regression, parameter optimization per user | 📐 Math & Logic | 📝 Planned |
| **03** | [Collaborative Filtering Algorithm](#03-collaborative-filtering-algorithm) | Joint optimization, latent factors $w^{(u)}$, $x^{(i)}$, simultaneous gradient descent | 📐 Math & Algo | 📝 Planned |
| **04** | [Binary Labels: Favs, Likes, and Clicks](#04-binary-labels-favs-likes-and-clicks) | Implicit feedback, logistic loss, binary cross-entropy formulation | 📐 Formulation | 📝 Planned |
| **05** | [Mean Normalization](#05-mean-normalization) | Cold-start mitigation, unrated items, baseline offset adjustment | 💡 Intuition | 📝 Planned |
| **06** | [TensorFlow Implementation of Collaborative Filtering](#06-tensorflow-implementation-of-collaborative-filtering) | Vectorization, `tf.Variable`, AutoDiff with `tf.GradientTape`, custom loss | 💻 Code / TF | 📝 Planned |
| **07** | [Finding Related Items](#07-finding-related-items) | Latent space distance, Euclidean distance $\|x^{(i)} - x^{(k)}\|^2$, Cosine similarity | 📐 Math & Logic | 📝 Planned |
| **08** | [Collaborative Filtering vs. Content-Based Filtering](#08-collaborative-filtering-vs-content-based-filtering) | Comparative trade-off matrix, cold-start vulnerability, scaling properties | ⚖️ Analysis | 📝 Planned |
| **09** | [Deep Learning for Content-Based Filtering](#09-deep-learning-for-content-based-filtering) | Two-tower neural architectures, user embeddings $v_u$, item embeddings $v_m$ | 🧠 Deep Learning | 📝 Planned |
| **10** | [Recommending from a Large Catalogue](#10-recommending-from-a-large-catalogue) | Two-stage pipeline: Retrieval (Candidate Generation) + Ranking, Vector Indexing | 🚀 Scale & Sys | 📝 Planned |
| **11** | [Ethical Use of Recommender Systems](#11-ethical-use-of-recommender-systems) | Filter bubbles, engagement traps, algorithmic bias, fairness, transparency | 🛡️ Ethics | 📝 Planned |
| **12** | [TensorFlow Implementation of Content-Based Filtering](#12-tensorflow-implementation-of-content-based-filtering) | Two-tower network in Keras/TF, dot-product output layer, evaluation | 💻 Code / TF | 📝 Planned |
| **13** | [Reducing the Number of Features (Optional)](#13-reducing-the-number-of-features-optional) | High-dimensionality curse, compression, variance preservation intuition | 🔍 Extension | 📝 Planned |
| **14** | [PCA Algorithm (Optional)](#14-pca-algorithm-optional) | Covariance matrix, projection, eigenvalue decomposition, singular value decomposition | 📐 Math / Algo | 📝 Planned |
| **15** | [PCA in Code (Optional)](#15-pca-in-code-optional) | From-scratch NumPy implementation, Scikit-Learn comparison, visualization | 💻 Code | 📝 Planned |

---

## 📖 Module-by-Module Detailed Breakdown

### Phase 1: Problem Formulation & Feature-Driven Baselines

#### [01. Making Recommendations](01_making_recommendations.md)
- **Intuition**: Why recommendations are distinct from standard supervised classification and regression.
- **Formulation**: The user-item interaction matrix $R$, where $r(i,j) = 1$ if user $j$ rated item $i$, and $y^{(i,j)}$ is the rating value.
- **Challenges**: Sparsity (typically $>99\%$ sparse), scale, and the cold-start dilemma.

#### [02. Using Per-Item Features](02_using_per-item-features.md)
- **Concept**: Leveraging item metadata $x^{(i)}$ (e.g., romance vs. action score) to learn personalized user preference vectors $w^{(j)}$ and bias $b^{(j)}$.
- **Model**: Predicted rating $\hat{y}^{(i,j)} = w^{(j)} \cdot x^{(i)} + b^{(j)}$.
- **Cost Function**: Regularized Mean Squared Error (MSE) optimized across rated items for each user independently.

---

### Phase 2: Collaborative Filtering & Latent Factor Models

#### [03. Collaborative Filtering Algorithm](03_collaborative_filtering_algorithm.md)
- **The Core Breakthrough**: What if we don't have explicit item features? We can learn user preferences $w^{(j)}$ and item features $x^{(i)}$ *simultaneously*.
- **Unified Objective Function**:
  $$J(w, b, x) = \frac{1}{2} \sum_{(i,j): r(i,j)=1} \left( w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)} \right)^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2$$
- **Optimization**: Gradient updates alternating or running jointly over parameter tensors.

#### [04. Binary Labels: Favs, Likes, and Clicks](04_binary_labels_favs_likes_and_clicks.md)
- **Implicit Feedback Reality**: Real-world platforms rarely collect 1-5 star ratings; interactions are binary (click, like, favorite, watch $>30\text{s}$).
- **Probabilistic Formulation**: Predicting $P(y^{(i,j)}=1) = g(w^{(j)} \cdot x^{(i)} + b^{(j)})$, where $g(z) = \frac{1}{1 + e^{-z}}$.
- **Loss Function**: Binary Cross-Entropy applied to interaction signals.

#### [05. Mean Normalization](05_mean_normalization.md)
- **The Cold-User Problem**: For a new user with zero ratings, standard regularization shrinks $w^{(j)} \to 0$, predicting zero for everything.
- **Normalization Strategy**: Subtract the mean rating $\mu_i$ for each item, train on normalized matrix, and predict $\hat{y}^{(i,j)} = w^{(j)} \cdot x^{(i)} + b^{(j)} + \mu_i$.
- **Outcome**: A new user with no history is defaulted to average item ratings instead of zeros.

#### [06. TensorFlow Implementation of Collaborative Filtering](06_tensorflow_implementation_of_collaborative_filtering.md)
- **Hands-on Lab**: Building collaborative filtering from scratch in TensorFlow.
- **Core Techniques**:
  - Vectorized loss computation using boolean masking (`tf.gather_nd` / element-wise multiplication).
  - Custom training loops using `tf.GradientTape()`.
  - Gradient optimization with `tf.keras.optimizers.Adam`.

#### [07. Finding Related Items](07_finding_related_items.md)
- **Geometric Latent Space**: Once features $x^{(i)}$ are learned, items reside in an $n$-dimensional latent semantic space.
- **Similarity Metrics**:
  - Squared distance: $\|x^{(i)} - x^{(k)}\|^2 = \sum_{l=1}^n (x_l^{(i)} - x_l^{(k)})^2$
  - Cosine similarity: $\cos(\theta) = \frac{x^{(i)} \cdot x^{(k)}}{\|x^{(i)}\| \|x^{(k)}\|}$
- **Use Cases**: "Users who liked this also liked..." and automated playlist / related video carousels.

---

### Phase 3: Content-Based Filtering & Modern Deep Architectures

#### [08. Collaborative Filtering vs. Content-Based Filtering](08_collaborative_filtering_vs_content-based_filtering.md)
- **Deep-Dive Comparative Analysis**:
  - **Collaborative Filtering**: Discovers serendipitous items; struggles with cold start; doesn't require domain features.
  - **Content-Based Filtering**: Excellent cold-start handling for new items; transparent and explainable; struggles to recommend outside user's current niche.
- **Decision Matrix**: When to choose which approach in production.

#### [09. Deep Learning for Content-Based Filtering](09_deep_learning_content-based_filtering.md)
- **Two-Tower Neural Network Architecture**:
  - **User Tower**: Deep Feedforward Network mapping user features (demographics, interaction history, device) $\to$ embedding vector $v_u \in \mathbb{R}^d$.
  - **Item Tower**: Deep Feedforward Network mapping item features (title embeddings, category, age, tags) $\to$ embedding vector $v_m \in \mathbb{R}^d$.
- **Prediction**: $\hat{y} = v_u \cdot v_m$.

#### [10. Recommending from a Large Catalogue](10_recommending_from_a_large_catalogue.md)
- **The Production Bottleneck**: Scoring millions of items with a neural network in real-time ($<50\text{ms}$) is computationally infeasible.
- **The Two-Stage Industry Standard Pipeline**:
  1. **Candidate Retrieval (Candidate Generation)**: Filter millions down to hundreds using fast Approximate Nearest Neighbors (ANN) vector search (e.g., Google ScaNN, FAISS, HNSW).
  2. **Scoring & Ranking**: Pass the candidate subset through a fine-grained, compute-heavy deep ranker.

#### [11. Ethical Use of Recommender Systems](11_ethical_use_of_recommender_systems.md)
- **Systemic Implications**:
  - **Filter Bubbles & Echo Chambers**: Reinforcement feedback loops polarizing users.
  - **Engagement vs. Well-being**: Clickbait optimization vs. long-term user satisfaction.
  - **Bias & Fairness**: Popularity bias, demographic bias, and creator exposure fairness.
- **Mitigation Techniques**: Exploration bonuses ($\epsilon$-greedy), diversification algorithms, and audit metrics.

#### [12. TensorFlow Implementation of Content-Based Filtering](12_tensorflow_implementation_of_content-based_filtering.md)
- **Hands-on Lab**: Building and training a complete Two-Tower neural network in TensorFlow/Keras.
- **Components**:
  - Multi-layer dense user tower and item tower with L2 normalization.
  - Custom Keras Model subclassing.
  - Fast inference simulation via vector dot product.

---

### Phase 4: Dimensionality Reduction & Latent Space Mastery (Extension)

#### [13. Reducing the Number of Features (Optional)](13_reducing_the_number_of_features_optional.md)
- **The Curse of Dimensionality**: Why sparse, high-dimensional spaces degrade distance metrics and increase computational overhead.
- **Compression & Visualization**: Projecting complex feature spaces down to 2D/3D for human interpretability.

#### [14. PCA Algorithm (Optional)](14_pca_algorithm_optional.md)
- **Mathematical Foundation**:
  - Feature normalization and zero-centering.
  - Covariance matrix calculation $\Sigma = \frac{1}{m} X^T X$.
  - Singular Value Decomposition (SVD) and Eigenvalue Decomposition.
  - Selecting principal components based on explained variance ratio.

#### [15. PCA in Code (Optional)](15_pca_in_code_optional.md)
- **Hands-on Lab**:
  - Implementing PCA from scratch using pure NumPy (`np.linalg.svd`).
  - Validation against `sklearn.decomposition.PCA`.
  - Visualizing high-dimensional recommender item embeddings in 2D latent space.

---

## 🛠️ Tech Stack & Environment Setup

To run the upcoming code labs and notebooks in this repository, set up your Python environment:

### 1. Clone the Repository
```bash
git clone https://github.com/TechGenDM/Recommender-Systems.git
cd Recommender-Systems
```

### 2. Create and Activate Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn jupyterlab
```

---

## 📁 Repository Directory Structure

As the modules are authored, the repository will be populated with comprehensive Markdown guides alongside runnable Jupyter notebooks:

```text
Recommender-Systems/
├── README.md                                          <- You are here (Curriculum Overview)
├── 01_making_recommendations.md                       <- Module 01 Notes
├── 02_using_per_item_features.md                      <- Module 02 Notes
├── 03_collaborative_filtering_algorithm.md            <- Module 03 Notes
├── 04_binary_labels_favs_likes_clicks.md              <- Module 04 Notes
├── 05_mean_normalization.md                           <- Module 05 Notes
├── 06_tensorflow_collaborative_filtering.md           <- Module 06 Notes & Code Walkthrough
├── 07_finding_related_items.md                        <- Module 07 Notes
├── 08_collaborative_vs_content_based_filtering.md     <- Module 08 Notes
├── 09_deep_learning_content_based_filtering.md        <- Module 09 Notes
├── 10_recommending_from_large_catalogue.md            <- Module 10 Notes
├── 11_ethical_use_of_recommender_systems.md           <- Module 11 Notes
├── 12_tensorflow_content_based_filtering.md           <- Module 12 Notes & Code Walkthrough
├── 13_reducing_number_of_features.md                  <- Module 13 Notes
├── 14_pca_algorithm.md                                <- Module 14 Notes
├── 15_pca_in_code.md                                  <- Module 15 Notes & Code Walkthrough
└── notebooks/                                         <- Executable Jupyter Notebooks (.ipynb)
```

---

## 🤝 Contributing

Contributions, issues, and feature suggestions are welcome! Whether it's correcting a typo, proposing an intuitive diagram, or adding a state-of-the-art retrieval benchmark:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingInsight`).
3. Commit your Changes (`git commit -m 'Add some AmazingInsight'`).
4. Push to the Branch (`git push origin feature/AmazingInsight`).
5. Open a Pull Request.

---

## 👨‍💻 Author & Maintainer

**Devasish Mishra**  
- **GitHub**: [@TechGenDM](https://github.com/TechGenDM)  
- **Repository**: [TechGenDM/Recommender-Systems](https://github.com/TechGenDM/Recommender-Systems)

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE) - feel free to use this content for educational, research, and self-study purposes.

⭐ **If you find this learning resource helpful, please consider starring the repository to support its reach!**