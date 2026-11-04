# 🚀 Setup Instructions

Complete guide to setting up the Credit Card Fraud Detection system.

---

## 📋 **Prerequisites**

### **Required Software**
- **Python 3.8+** (3.9 or 3.10 recommended)
- **pip** (Python package manager)
- **Git** (for cloning repository)

### **Optional Software**
- **Anaconda/Miniconda** (alternative to venv)
- **Jupyter** (for notebooks)
- **VS Code** or **PyCharm** (IDEs)

### **System Requirements**
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB free space
- **OS**: Windows, macOS, or Linux

---

## 📥 **Installation Steps**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

Or download and extract the ZIP file.

### **2. Create Virtual Environment**

#### **Option A: Using venv (Recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

#### **Option B: Using conda**

```bash
# Create conda environment
conda create -n fraud-detection python=3.9

# Activate environment
conda activate fraud-detection
```

### **3. Install Dependencies**

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Installation time**: ~5-10 minutes depending on internet speed.

### **4. Verify Installation**

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Test imports
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print('✅ All imports successful!')"
```

---

## 📊 **Data Setup**

### **Option 1: Download Real Dataset (Recommended)**

1. **Visit** Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud

2. **Download** `creditcard.csv` (144 MB)

3. **Place** in data directory:
   ```bash
   # Create data directory if it doesn't exist
   mkdir -p data
   
   # Move downloaded file
   mv ~/Downloads/creditcard.csv data/
   ```

4. **Verify**:
   ```bash
   ls -lh data/creditcard.csv
   # Should show ~144MB file
   ```

### **Option 2: Use Synthetic Data (No Download)**

The training script will automatically generate synthetic data if no dataset is found.

**No action required!** Just proceed to training.

---

## 🏃 **Quick Start**

### **1. Train Models**

```bash
# Train with synthetic data
python train.py

# Or train with real dataset (if downloaded)
python train.py
```

**Training time**: 
- Synthetic data (100K transactions): ~3-5 minutes
- Real dataset (280K transactions): ~10-15 minutes

**Output**:
- Trained models saved to `models/`
- Evaluation reports in `reports/`
- Visualizations in `reports/figures/`

### **2. Launch Web Interface**

```bash
streamlit run app/app.py
```

**Access**: Open browser to `http://localhost:8501`

### **3. Explore with Jupyter**

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/exploration.ipynb
```

---

## 🔧 **Configuration**

### **Environment Variables (Optional)**

Create a `.env` file in the project root:

```bash
# .env
DATA_PATH=data/creditcard.csv
MODEL_PATH=models/
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### **Streamlit Configuration (Optional)**

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
address = "localhost"

[theme]
primaryColor = "#00d4ff"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#ffffff"
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Issue: Import Error for XGBoost**

```bash
# Solution: Reinstall XGBoost
pip uninstall xgboost
pip install xgboost==2.0.0
```

#### **Issue: Streamlit not found**

```bash
# Solution: Reinstall Streamlit
pip install --upgrade streamlit
```

#### **Issue: Memory Error during training**

```python
# Solution: Use smaller sample size
# Edit train.py line 250:
trainer.load_and_prepare_data(sample_size=50000)  # Reduce from 100000
```

#### **Issue: Permission denied**

```bash
# Solution: Run with appropriate permissions
# Windows:
Right-click > Run as administrator

# macOS/Linux:
sudo python train.py
```

#### **Issue: Port 8501 already in use**

```bash
# Solution: Use different port
streamlit run app/app.py --server.port 8502
```

---

## 🧪 **Testing Installation**

### **Test 1: Check Dependencies**

```bash
python -c "
import pandas
import numpy
import sklearn
import xgboost
import imblearn
import streamlit
import matplotlib
import seaborn
print('✅ All dependencies OK!')
"
```

### **Test 2: Run Mini Training**

```python
# test_setup.py
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.evaluation import ModelEvaluator

print("✅ All modules imported successfully!")
print("🎉 Setup complete! You're ready to go!")
```

```bash
python test_setup.py
```

---

## 📚 **Next Steps**

1. ✅ **Read README.md** for project overview
2. ✅ **Run `python train.py`** to train models
3. ✅ **Launch `streamlit run app/app.py`** for web interface
4. ✅ **Explore `notebooks/exploration.ipynb`** for analysis
5. ✅ **Read CONTRIBUTING.md** if you want to contribute

---

## 💡 **Tips**

### **For Faster Training**
- Use SSD for faster file I/O
- Close other applications to free RAM
- Use `n_jobs=-1` to utilize all CPU cores

### **For Development**
```bash
# Install development dependencies
pip install black flake8 pytest

# Format code
black train.py

# Run tests
pytest tests/
```

### **For Production Deployment**
See detailed deployment guide in the main README.md

---

## 🆘 **Getting Help**

- **Documentation**: See README.md
- **Issues**: https://github.com/yourusername/fraud-detection/issues
- **Discussions**: https://github.com/yourusername/fraud-detection/discussions
- **Email**: your.email@example.com

---

## ✅ **Setup Checklist**

- [ ] Python 3.8+ installed
- [ ] Repository cloned/downloaded
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded (optional)
- [ ] Models trained (`python train.py`)
- [ ] Web app launched (`streamlit run app/app.py`)
- [ ] Jupyter notebook tested (optional)

**All done?** You're ready to detect fraud! 🎉

---

**Questions?** Open an issue on GitHub or contact the maintainers.
