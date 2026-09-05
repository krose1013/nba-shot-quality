#  NBA Expected Shot Quality (xFG%) & Spatial Heatmap Analysis

**Author:** Kylah Rose  
**Live App:** [Streamlit Dashboard](nba-shot-quality-tyexthmhzlb453kh83nosu.streamlit.app)  
**Repository:** [github.com/krose1013/nba-shot-quality](https://github.com/krose1013/nba-shot-quality)  

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)

---

##  Project Overview

The **NBA Expected Shot Quality (xFG%) Model** is an interactive spatial analytics tool designed to evaluate player and team shot selection efficiency. By training a machine learning model on **2025–26 NBA spatial shot-chart coordinates**, the application calculates the baseline probability of any shot going in based on distance, court location, and game context.

By comparing a player's actual Field Goal Percentage against their Expected Field Goal Percentage ($\text{xFG\%}$), the dashboard isolates pure **Shooting Over Expectation (ShoE)**—helping analysts identify elite shot-makers versus players who benefit from high-quality shot context.

---

##  Key Features

- **Automated Data Ingestion:** Pulls detailed spatial shot-chart coordinates ($X, Y$ location, distance, zone) dynamically via `nba_api` for the **2025–26 NBA season**.
- **Machine Learning Shot Model:** Trains a Logistic Regression classifier on distance and spatial positioning to predict baseline shot probability ($\text{xFG\%}$).
- **Shooting Over Expectation Metric (ShoE):** Calculates $\text{Actual FG\%} - \text{Expected FG\%}$ to evaluate player performance relative to shot difficulty.
- **Interactive Spatial Visualizations:** Allows users to toggle between scatter shot-charts (makes/misses) and 2D Kernel Density Estimate (KDE) volume heatmaps.
- **Automated Development Feedback:** Evaluates individual and team tendencies to highlight primary strength areas, development zones needing improvement, and highest-volume spots.

---

##  Mathematical Formulation & Feature Engineering

### 1. Expected Field Goal Percentage Model ($\text{xFG\%}$)
The baseline shot probability is modeled using a Logistic Regression sigmoid function trained on spatial features:

$$P(\text{Make}_i = 1) = \frac{1}{1 + e^{-z_i}}$$

Where $z_i$ represents the linear combination of spatial shot features:

$$z_i = \beta_0 + \beta_1 (\text{Shot Distance}_i) + \beta_2 (\text{Loc}_X) + \beta_3 (\text{Loc}_Y) + \beta_4 (\text{Period}_i) + \beta_5 (\text{Time Remaining}_i)$$

---

### 2. Shooting Over Expectation (ShoE)
To measure whether a player outperforms or underperforms the average league efficiency given their exact shot diet, **ShoE** is calculated as:

$$\text{ShoE} = (\text{Actual FG\%} - \text{Expected FG\%}) \times 100$$

* **Positive ShoE ($> 0$):** Indicates elite shot-making ability (converting tough shots at an above-average rate).
* **Negative ShoE ($< 0$):** Indicates poor shot efficiency relative to the difficulty of the opportunities taken.

---

##  Project Architecture

```text
nba-shot-quality/
├── data_loader.py          # Downloads 2025-26 shot chart data from NBA API
├── train_model.py          # Trains Logistic Regression model & computes xFG%
├── app.py                  # Interactive Streamlit dashboard & heatmap UI
├── shot_data.csv           # Raw NBA shot chart dataset
├── shot_data_with_xfg.csv  # Processed dataset with xFG% predictions
├── shot_model.pkl          # Saved trained Scikit-Learn model
└── requirements.txt        # Production dependencies (streamlit, scikit-learn, etc.)
