# BIA — Business Intelligence & Analytics

Machine learning and data science learning projects.

## Projects

### 1. MachineLearningIntroduction
**Simple Linear Regression Model for Housing Price Prediction**

A comprehensive introduction to the machine learning pipeline covering data exploration, preprocessing, model training, and evaluation.

- **Techniques:** Exploratory Data Analysis (EDA), Data Preprocessing, Feature Scaling, Linear Regression, Model Evaluation
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Dataset:** 128 housing records (125 after cleaning)
- **Performance:** R² = 0.7928 (79.28% variance explained), MAE = $9,165, RMSE = $11,548

**Quick Start:**
```bash
cd MachineLearningIntroduction
python main.py
```

**Full Documentation:** [MachineLearningIntroduction/README.md](MachineLearningIntroduction/README.md)

---

## Getting Started

### Prerequisites
- Python 3.14+
- Git
- GitHub account

### Setup GitHub Credentials

1. **Create a GitHub Personal Access Token:**
   - Go to [GitHub Settings → Personal Access Tokens](https://github.com/settings/tokens)
   - Click **Generate new token**
   - Select scope: `repo` (full control of private repositories)
   - Copy the token (you won't be able to see it again!)

2. **Configure local `.env` file:**
   ```bash
   # Copy template
   cp .env.example .env
   
   # Edit .env with your credentials
   # (Windows) Notepad .env
   # (Linux/Mac) nano .env
   ```

3. **Fill in `.env`:**
   ```bash
   GITHUB_USERNAME=your_username
   GITHUB_TOKEN=your_personal_access_token
   GITHUB_REPO=BIA
   ```

**⚠️ Security:** The `.env` file is in `.gitignore` and will NEVER be committed to GitHub. Keep your credentials safe!

### Clone & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/BIA.git
cd BIA

# Navigate to a project
cd MachineLearningIntroduction

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python main.py
```

---

## Project Structure

```
BIA/
├── README.md                          (this file)
├── LICENSE                            (MIT License)
├── .gitignore                         (Python .gitignore)
│
└── MachineLearningIntroduction/
    ├── README.md                      (Project documentation)
    ├── requirements.txt               (Project dependencies)
    ├── main.py                        (Entry point)
    ├── data_loader.py                 (Data loading & inspection)
    ├── preprocessing.py               (Feature engineering & scaling)
    ├── eda.py                         (Exploratory data analysis)
    ├── model.py                       (Model training & evaluation)
    │
    └── data/
        └── housing_dataset.csv        (Dataset)
```

---

## Future Projects

Additional ML/data science projects will be added to this repository:
- Advanced Regression Models (Polynomial, Ridge, Lasso)
- Classification Models (Logistic Regression, Decision Trees)
- Time Series Forecasting
- Clustering & Unsupervised Learning
- Natural Language Processing
- Deep Learning

---

## Technologies

- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Python 3.14+, Virtual Environment

---

## Learning Objectives

This repository demonstrates:
- ✓ End-to-end ML pipeline development
- ✓ Data exploration & visualization
- ✓ Data preprocessing & feature engineering
- ✓ Model training & evaluation
- ✓ Metric interpretation
- ✓ Code documentation & organization
- ✓ Best practices in machine learning

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

This is a personal learning repository. Feedback and suggestions are welcome!

---

## Contact

For questions or discussions about these projects, feel free to reach out.

---

**Last Updated:** 2026-07-20
