# MACHINE-LEARNING
* **1.SIMPLE LINEAR REGRESSION:-**

### **Simple Linear Regression (SLR)**  

Simple Linear Regression is a fundamental machine learning algorithm used for predicting a continuous dependent variable (\(Y\)) based on a single independent variable (\(X\)). It establishes a linear relationship between \(X\) and \(Y\) using the equation:  

\[
Y = mX + c
\]

Where:  
- \( Y \) = Dependent variable (target)  
- \( X \) = Independent variable (feature)  
- \( m \) = Slope of the line (coefficient)  
- \( c \) = Intercept (constant)  

### **Steps to Perform Simple Linear Regression**  
1. **Import Libraries**  
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error, r2_score
   ```

2. **Load Dataset**  
   ```python
   df = pd.read_csv("data.csv")  # Load dataset
   X = df[['feature_column']]    # Independent variable
   Y = df['target_column']       # Dependent variable
   ```

3. **Split Data into Training and Testing Sets**  
   ```python
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
   ```

4. **Train the Model**  
   ```python
   model = LinearRegression()
   model.fit(X_train, Y_train)
   ```

5. **Make Predictions**  
   ```python
   Y_pred = model.predict(X_test)
   ```

6. **Evaluate the Model**  
   ```python
   print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}")
   print(f"R² Score: {r2_score(Y_test, Y_pred)}")
   ```

7. **Visualize the Results**  
   ```python
   plt.scatter(X_test, Y_test, color='blue', label='Actual Data')
   plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Regression Line')
   plt.xlabel("Independent Variable (X)")
   plt.ylabel("Dependent Variable (Y)")
   plt.title("Simple Linear Regression")
   plt.legend()
   plt.show()
   ```

### **Use Case Example**  
You recently built an SLR model to analyze the relationship between **CGPA and package (LPA)**. In this case:  
- **CGPA** is the independent variable (\(X\))  
- **Package (LPA)** is the dependent variable (\(Y\))  
- The model predicts how a student's CGPA influences their salary package.
  ```
  * **1.MULTIPLE LINEAR REGRESSION:-** 


