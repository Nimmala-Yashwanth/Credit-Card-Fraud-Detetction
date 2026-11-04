# 📊 Dataset Directory

This directory contains the credit card transaction dataset used for fraud detection.

---

## 📥 **Download Dataset**

### **Option 1: Kaggle (Recommended)**

1. **Visit**: https://www.kaggle.com/mlg-ulb/creditcardfraud

2. **Download** `creditcard.csv` (144 MB)

3. **Place** the file in this directory:
   ```
   fraud-detection/data/creditcard.csv
   ```

### **Option 2: Use Synthetic Data**

If you don't have access to the dataset, the training script will automatically generate 100,000 synthetic transactions with realistic fraud patterns.

**No download required!** Just run:
```bash
python train.py
```

---

## 📊 **Dataset Information**

**Name**: Credit Card Fraud Detection  
**Source**: Worldline and Machine Learning Group (ULB)  
**Size**: 150 MB  
**Transactions**: 284,807  
**Fraud Cases**: 492 (0.173%)  
**Features**: 30 (28 PCA + Time + Amount)  

### **Feature Description**

| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed since first transaction |
| `V1-V28` | PCA components (anonymized) |
| `Amount` | Transaction amount in Euros |
| `Class` | 0 = Normal, 1 = Fraud |

---

## 🔒 **Data Privacy**

The dataset contains **anonymized** transaction data. The original features have been transformed using PCA to protect customer privacy.

**Do not attempt to:**
- Re-identify individuals
- Reverse PCA transformations
- Use for unauthorized purposes

This data is for **research and educational purposes only**.

---

## 📚 **Citation**

If you use this dataset in your research, please cite:

```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. 
Calibrating Probability with Undersampling for Unbalanced Classification. 
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
```

---

## ✅ **Verification**

After downloading, verify the dataset:

```bash
# Check file exists
ls -lh creditcard.csv

# Expected output:
# -rw-r--r-- 1 user user 144M creditcard.csv

# View first few lines
head -5 creditcard.csv
```

**Expected columns:**
```
Time,V1,V2,V3,...,V28,Amount,Class
```

---

## 🚫 **.gitignore**

The `creditcard.csv` file is excluded from Git to avoid:
- Pushing large files to GitHub
- Violating data usage terms
- Repository bloat

**Always download fresh from Kaggle!**

---

## 📁 **Directory Contents**

```
data/
├── .gitkeep              # Keeps directory in Git
├── README.md             # This file
└── creditcard.csv        # Download separately (not in Git)
```

---

## 🤝 **Alternative Datasets**

If you want to experiment with other fraud detection datasets:

1. **IEEE-CIS Fraud Detection**: https://www.kaggle.com/c/ieee-fraud-detection
2. **Synthetic Financial Datasets**: https://github.com/IBM/TabFormer
3. **PaySim Mobile Money Dataset**: https://www.kaggle.com/ntnu-testimon/paysim1

---

## 💡 **Tips**

- **Storage**: Ensure you have at least 500 MB free space
- **Performance**: SSD recommended for faster loading
- **Memory**: 4GB+ RAM recommended for full dataset
- **Subset**: You can use a smaller subset for faster experimentation

---

**Questions?** See the main [README.md](../README.md) or open an issue on GitHub.
