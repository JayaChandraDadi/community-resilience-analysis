# Community Resilience Analysis Using Human Mobility Data

This project analyzes the resilience of communities in Port Arthur, Texas, during natural disasters using SafeGraph mobility data and methodologies from two research papers. The analysis is implemented in Python and follows a two-part structure inspired by:

1. **Paper 1**: *Hong et al., 2021 - Measuring inequality in community resilience to natural disasters*
2. **Paper 2**: *Community resilience to wildfires: A network analysis approach by utilizing human mobility data*

---

## 🔍 Project Objectives

- Identify disruptions to human mobility due to natural disasters.
- Measure community resilience using the **Resilience Triangle** and **centrality-based metrics**.
- Compare resilience across Census Block Groups (CBGs).
- Visualize mobility and resilience patterns over time.

---

## 📁 Project Structure

communityresilience/
│
├── data/ # Contains input datasets (e.g., .csv, .RData)
│
├── paper1_rcn/ # Code based on Paper 1
│ └── paper1_rcn_analysis.py # Reads RData and computes RCN index
│
├── paper2_triangle/ # Code based on Paper 2
│ └── paper2_resilience_triangle.py # Network-based resilience triangle analysis
│
├── notebooks/ # Jupyter Notebooks for testing and EDA
│ ├── ra paper 1.ipynb
│ └── ra paper 2.ipynb
│
├── run_all.py # Runs both paper analyses
└── README.md # You’re here!

---

## 📚 Conceptual Background

### 📘 Paper 1: Hong et al. (2021)
- **Focus**: Quantifying community resilience with a **Resilience Capacity Network (RCN)** index.
- **Steps**:
  - Read mobility metrics from RData file.
  - Normalize and scale features using `StandardScaler`.
  - Calculate the RCN index for each CBG based on feature weights.

### 📘 Paper 2: Wildfire Community Resilience
- **Focus**: Uses mobility data to construct monthly CBG mobility networks.
- **Steps**:
  - Create a graph for each month using SafeGraph data.
  - Compute **closeness centrality** for each CBG.
  - Identify disruptions and calculate the **Resilience Triangle** (Drop % and Recovery %).
  - Perform **DTW clustering** to group CBGs by resilience pattern.
  - Apply regression to explore factors influencing resilience.

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/JayaChandraDadi/community-resilience-analysis.git
cd communityresilience
How to Run
1. Run Paper 1 (RCN Analysis)
python paper1_rcn/paper1_rcn_analysis.py
.2 Run Paper 2 (Triangle Analysis)
python paper2_triangle/paper2_resilience_triangle.py
3. Run all at once
python run_all.py
Outputs
RCN index per CBG (Paper 1)

Resilience triangle metrics per CBG (Paper 2)

Clustering plots and regression analysis

CSV files summarizing resilience scores
