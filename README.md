# 🏠 Boston House Price Prediction

A machine learning project that predicts house prices in Boston using regression algorithms with a Flask web application interface.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [How to Use the Application](#how-to-use-the-application)
- [Making Predictions](#making-predictions)
- [Models & Performance](#models--performance)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project implements a **machine learning regression model** that predicts the median home prices in Boston. The model is trained on historical data from 1978 and packaged with a user-friendly Flask web application, allowing users to input property characteristics and receive price predictions instantly.

**Key Highlights:**
- ✅ Multiple regression algorithms tested and evaluated
- ✅ Interactive web interface for easy predictions
- ✅ Pre-trained, serialized model (pickle files)
- ✅ Data preprocessing with StandardScaler
- ✅ RESTful API endpoint for programmatic predictions

---

## 📊 Dataset Information

The **Boston Housing Dataset** contains:
- **506 records** of suburban homes from Boston
- **14 attributes** describing various property characteristics
- **Collected in 1978** - historical real estate data

### Dataset Attributes:

| Attribute | Description |
|-----------|-------------|
| **CRIM** | Per capita crime rate by town |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft. |
| **INDUS** | Proportion of non-retail business acres per town |
| **CHAS** | Charles River dummy variable (1 = bounds river, 0 = otherwise) |
| **NOX** | Nitric oxides concentration (parts per 10 million) |
| **RM** | Average number of rooms per dwelling |
| **AGE** | Proportion of owner-occupied units built prior to 1940 |
| **DIS** | Weighted distances to five Boston employment centers |
| **RAD** | Index of accessibility to radial highways |
| **TAX** | Full-value property-tax rate per $10,000 |
| **PTRATIO** | Pupil-teacher ratio by town |
| **BLACK** | 1000(Bk - 0.63)² where Bk is proportion of blacks by town |
| **LSTAT** | % lower status of the population |
| **MEDV** | **Median value of owner-occupied homes in $1000's** (TARGET) |

**Data Source:** [Boston Housing Dataset on Kaggle](https://www.kaggle.com/puxama/bostoncsv)

---

## ✨ Features

- **Multiple Regression Algorithms:**
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Extra Trees
  - XGBoost

- **Web Interface:**
  - User-friendly form to input property features
  - Real-time price predictions
  - Professional UI with HTML/CSS templates

- **API Endpoint:**
  - JSON-based API for programmatic predictions
  - Integration-ready for other applications

- **Data Preprocessing:**
  - StandardScaler for feature normalization
  - Train-test split (80-20 ratio)
  - Feature correlation analysis

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/irritatednishant/boston_house_prediction.git
cd boston_house_prediction
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

The application will start at `http://localhost:5000`

---

## 📁 Project Structure

```
boston_house_prediction/
│
├── Boston_house_prediction.ipynb    # Jupyter notebook with model training & analysis
├── Boston (1).csv                   # Dataset file
├── app.py                           # Flask application
├── regression_model.pkl             # Trained regression model (serialized)
├── scaler_model.pkl                 # StandardScaler object (serialized)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── LICENSE                          # Apache License 2.0
│
└── templates/
    └── home.html                    # Web interface HTML template
```

---

## 🎮 How to Use the Application

### Method 1: Web Interface

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Fill in the form** with property characteristics:
   - Crime rate (CRIM)
   - Zoning information (ZN)
   - Industrial acres (INDUS)
   - River proximity (CHAS)
   - NOX concentration (NOX)
   - Number of rooms (RM)
   - Property age (AGE)
   - Distance to employment (DIS)
   - Highway accessibility (RAD)
   - Property tax (TAX)
   - Pupil-teacher ratio (PTRATIO)
   - Black population ratio (BLACK)
   - Lower status percentage (LSTAT)

4. **Click "Predict"** to get the estimated house price

### Method 2: API Endpoint

**Endpoint:** `POST /predict_api`

**Request Format (JSON):**
```json
{
  "data": {
    "crim": 0.02731,
    "zn": 0.0,
    "indus": 7.07,
    "chas": 0,
    "nox": 0.469,
    "rm": 6.421,
    "age": 78.9,
    "dis": 4.9671,
    "rad": 2,
    "tax": 242,
    "ptratio": 17.8,
    "black": 396.90,
    "lstat": 9.14
  }
}
```

**Example using cURL:**
```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "crim": 0.02731, "zn": 0.0, "indus": 7.07,
      "chas": 0, "nox": 0.469, "rm": 6.421,
      "age": 78.9, "dis": 4.9671, "rad": 2,
      "tax": 242, "ptratio": 17.8, "black": 396.90,
      "lstat": 9.14
    }
  }'
```

**Response:**
```json
21.6
```

---

## 📈 Making Predictions

### Understanding the Prediction

The model outputs a **predicted median house value in thousands of dollars**.

**Example:**
- If the model returns `24.5`, the predicted house price is **$24,500**
- If the model returns `35.2`, the predicted house price is **$35,200**

### Tips for Better Predictions

1. **Ensure accurate input values** - Use real-world data from property assessments
2. **Keep values within dataset ranges** - The model works best with values similar to the training data
3. **Consider seasonal factors** - Historical data from 1978 may not reflect modern prices
4. **Multiple iterations** - Try different scenarios to understand how each feature affects price

---

## 🤖 Models & Performance

### Algorithms Tested

| Algorithm | Description |
|-----------|-------------|
| **Linear Regression** | Simple baseline model with strong performance |
| **Decision Tree** | Captures non-linear relationships |
| **Random Forest** | Ensemble of decision trees, reduces overfitting |
| **Extra Trees** | Similar to Random Forest with randomized splits |
| **XGBoost** | Gradient boosting, highly optimized |

### Performance Metrics

- **Mean Squared Error (MSE):** 10.0
- **Model Accuracy:** Excellent predictions on test set
- **Training Data:** 404 samples (80%)
- **Test Data:** 102 samples (20%)

### Feature Importance

Based on correlation analysis, the most important features affecting house prices are:
1. **RM** (number of rooms) - Positive correlation (0.70)
2. **LSTAT** (lower status %) - Negative correlation (-0.74)
3. **PTRATIO** (pupil-teacher ratio) - Negative correlation (-0.51)
4. **NOX** (pollution) - Negative correlation (-0.43)
5. **DIS** (distance to employment) - Positive correlation (0.25)

---

## 🛠️ Technologies Used

### Core Libraries
- **Flask** (3.1.2) - Web framework
- **Scikit-learn** (1.7.2) - Machine learning algorithms
- **Pandas** (2.3.3) - Data manipulation
- **NumPy** (2.3.4) - Numerical computing
- **Matplotlib** (3.10.7) - Data visualization

### Additional Tools
- **Gunicorn** (23.0.0) - WSGI HTTP server for production
- **Joblib** (1.5.2) - Model serialization
- **Jinja2** (3.1.6) - Template engine

### Development Environment
- Python 3.7+
- Jupyter Notebook for analysis
- Git for version control

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/YourFeature`
3. **Make your changes** and commit: `git commit -m 'Add YourFeature'`
4. **Push to the branch:** `git push origin feature/YourFeature`
5. **Submit a Pull Request**

### Ideas for Improvement
- Add more recent Boston housing data
- Implement hyperparameter tuning
- Add data validation and error handling
- Create a mobile-friendly interface
- Deploy to cloud platform (Heroku, AWS, etc.)

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Nishant** - [GitHub Profile](https://github.com/irritatednishant)

---

## 📞 Support & Questions

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/irritatednishant/boston_house_prediction/issues)
- Create a [Discussion](https://github.com/irritatednishant/boston_house_prediction/discussions)

---

## 🎓 Learning Resources

This project demonstrates:
- Machine learning with scikit-learn
- Model serialization (pickle)
- Flask web development
- RESTful API design
- Data preprocessing and normalization
- Train-test data splitting
- Model evaluation and comparison

---

**Happy Predicting! 🚀**
