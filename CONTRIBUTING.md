# 🤝 Contributing to Credit Card Fraud Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## 📋 **Table of Contents**

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submission Guidelines](#submission-guidelines)
- [Review Process](#review-process)

---

## 📜 **Code of Conduct**

### **Our Pledge**

We pledge to make participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### **Our Standards**

**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior:**
- Harassment, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

---

## 💡 **How Can I Contribute?**

### **1. Reporting Bugs**

**Before submitting:**
- Check existing issues to avoid duplicates
- Use the latest version
- Test with synthetic data to isolate the issue

**When reporting:**
- Use a clear, descriptive title
- Describe exact steps to reproduce
- Provide error messages and stack traces
- Include your environment (OS, Python version, package versions)

**Template:**
```markdown
**Bug Description:**
Brief description of the bug.

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**Environment:**
- OS: Windows 10 / macOS 13 / Ubuntu 22.04
- Python: 3.9.7
- Package versions: (pip list)

**Error Message:**
```
Full error stack trace
```
```

### **2. Suggesting Enhancements**

**Enhancement ideas:**
- New ML models (LightGBM, CatBoost, Neural Networks)
- Additional feature engineering techniques
- Performance optimizations
- UI/UX improvements in Streamlit app
- Better visualizations
- API endpoints for production
- Docker containerization
- Cloud deployment guides

**Template:**
```markdown
**Enhancement Title:**
Clear, concise title.

**Problem/Motivation:**
What problem does this solve?

**Proposed Solution:**
Detailed description of the enhancement.

**Alternatives Considered:**
Other solutions you've thought about.

**Additional Context:**
Any other relevant information.
```

### **3. Contributing Code**

**Areas for contribution:**
- Bug fixes
- Feature implementations
- Documentation improvements
- Test coverage
- Performance optimizations
- Code refactoring

---

## 🔧 **Development Setup**

### **1. Fork and Clone**

```bash
# Fork the repository on GitHub

# Clone your fork
git clone https://github.com/YOUR-USERNAME/fraud-detection.git
cd fraud-detection

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/fraud-detection.git
```

### **2. Create Branch**

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

### **3. Install Development Dependencies**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install project + dev dependencies
pip install -r requirements.txt
pip install black flake8 pytest pytest-cov mypy
```

### **4. Make Changes**

```bash
# Make your changes
# ...

# Format code
black .

# Run linter
flake8 .

# Run tests
pytest tests/
```

---

## 📝 **Coding Standards**

### **Python Style**

We follow **PEP 8** with some modifications:

```python
# Use Black for formatting
# Line length: 88 characters (Black default)
# Use type hints where appropriate

from typing import List, Dict, Tuple, Optional
import pandas as pd

def process_transactions(
    transactions: pd.DataFrame,
    threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Process transactions for fraud detection.
    
    Args:
        transactions: DataFrame with transaction data
        threshold: Fraud probability threshold
        
    Returns:
        Tuple of (predictions DataFrame, metrics dictionary)
    """
    # Implementation
    ...
```

### **Documentation**

**Docstrings:**
```python
def function_name(param1: str, param2: int) -> bool:
    """
    One-line summary.
    
    Detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param2 is negative
        
    Example:
        >>> function_name("test", 5)
        True
    """
    pass
```

**Comments:**
```python
# Good: Explains WHY, not WHAT
# Apply SMOTE to balance training data
X_resampled, y_resampled = smote.fit_resample(X, y)

# Bad: States the obvious
# Resample X and y using smote
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### **Naming Conventions**

```python
# Classes: PascalCase
class FraudDetector:
    pass

# Functions and variables: snake_case
def calculate_fraud_score(transaction_amount: float) -> float:
    fraud_threshold = 0.5
    return score

# Constants: UPPER_SNAKE_CASE
MAX_TRANSACTION_AMOUNT = 10000
DEFAULT_MODEL_PATH = "models/"

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### **File Organization**

```python
# 1. Standard library imports
import os
import sys
from datetime import datetime

# 2. Third-party imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 3. Local imports
from src.preprocessing import DataPreprocessor
from src.evaluation import ModelEvaluator
```

---

## 📤 **Submission Guidelines**

### **1. Commit Messages**

Use conventional commits format:

```bash
# Format: <type>(<scope>): <subject>

# Types:
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# style: Code formatting (no logic change)
# refactor: Code refactoring
# test: Adding/updating tests
# chore: Maintenance tasks

# Examples:
git commit -m "feat(models): add LightGBM classifier"
git commit -m "fix(preprocessing): handle missing values in Amount column"
git commit -m "docs(README): update installation instructions"
git commit -m "refactor(evaluation): extract plotting functions"
```

### **2. Pull Request Process**

1. **Ensure your code:**
   - Passes all tests (`pytest`)
   - Is formatted (`black .`)
   - Has no linting errors (`flake8 .`)
   - Includes tests for new functionality
   - Updates documentation as needed

2. **Create Pull Request:**
   ```markdown
   **Description:**
   Clear description of changes.
   
   **Related Issue:**
   Fixes #123
   
   **Type of Change:**
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   **Testing:**
   - [ ] All existing tests pass
   - [ ] Added new tests for changes
   - [ ] Tested locally with synthetic data
   - [ ] Tested with real dataset (if applicable)
   
   **Checklist:**
   - [ ] Code follows project style guidelines
   - [ ] Self-reviewed the code
   - [ ] Commented hard-to-understand areas
   - [ ] Updated documentation
   - [ ] No new warnings generated
   ```

3. **Wait for Review:**
   - Maintainers will review within 3-5 business days
   - Address feedback promptly
   - Keep PR focused on single issue/feature

---

## 🔍 **Review Process**

### **What Reviewers Look For**

1. **Functionality:**
   - Does it work as intended?
   - Are edge cases handled?
   - Are there any bugs?

2. **Code Quality:**
   - Is it readable and maintainable?
   - Follows coding standards?
   - Appropriate comments and documentation?

3. **Testing:**
   - Are there tests?
   - Do tests cover edge cases?
   - Do all tests pass?

4. **Performance:**
   - Is it efficient?
   - Any unnecessary computations?
   - Memory usage acceptable?

### **Response Time**

- **Initial review**: Within 3-5 business days
- **Follow-up reviews**: Within 1-2 business days
- **Merge**: After approval and passing all checks

---

## 🧪 **Testing Guidelines**

### **Writing Tests**

```python
# tests/test_preprocessing.py
import pytest
import pandas as pd
from src.preprocessing import DataPreprocessor

def test_clean_data_removes_duplicates():
    """Test that clean_data removes duplicate rows."""
    # Arrange
    df = pd.DataFrame({
        'Amount': [100, 100, 200],
        'Class': [0, 0, 1]
    })
    preprocessor = DataPreprocessor()
    
    # Act
    result = preprocessor.clean_data(df)
    
    # Assert
    assert len(result) == 2

def test_clean_data_handles_missing_values():
    """Test that clean_data handles missing values."""
    # Arrange
    df = pd.DataFrame({
        'Amount': [100, None, 200],
        'Class': [0, 0, 1]
    })
    preprocessor = DataPreprocessor()
    
    # Act
    result = preprocessor.clean_data(df)
    
    # Assert
    assert result['Amount'].isna().sum() == 0
```

### **Running Tests**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run specific test
pytest tests/test_preprocessing.py::test_clean_data_removes_duplicates
```

---

## 📊 **Performance Guidelines**

### **Optimization Tips**

1. **Vectorization:**
   ```python
   # Good: Use pandas/numpy vectorization
   df['Amount_log'] = np.log1p(df['Amount'])
   
   # Avoid: Iterating rows
   for idx, row in df.iterrows():
       df.at[idx, 'Amount_log'] = np.log1p(row['Amount'])
   ```

2. **Memory Management:**
   ```python
   # Use appropriate data types
   df['Class'] = df['Class'].astype('int8')  # Instead of int64
   
   # Delete unused DataFrames
   del df_temp
   gc.collect()
   ```

3. **Profiling:**
   ```python
   # Use cProfile for profiling
   import cProfile
   cProfile.run('train_model()')
   ```

---

## 🏆 **Recognition**

Contributors will be:
- Listed in README.md
- Mentioned in release notes
- Added to CONTRIBUTORS.md (coming soon)

---

## 📞 **Questions?**

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bugs and features
- **Email**: your.email@example.com

---

**Thank you for contributing!** 🎉

Every contribution, no matter how small, makes this project better.
