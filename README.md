# Linear Regression | Normal Equation
A linear regression model built **from scratch in Python** using the normal equation, no gradient descent, no scikit-learn for the core model.

## What it does
- Implements **linear regression** using the closed-form normal equation
- Adds a bias column manually and solves for weights in one matrix operation
- Tests the model on the **California housing dataset**
- Compares results against scikit-learn's `LinearRegression` using **R² score**

## Math behind it
Instead of iterating with gradient descent, the normal equation solves for the optimal weights directly by minimizing the sum of squared errors:

$$\theta = (X^T X)^{-1} X^T y$$

Where:
- `X` is the feature matrix with a column of 1s prepended (for the bias/intercept term)
- `y` is the target vector
- `θ` contains the intercept (`θ₀`) and coefficients (`θ₁, θ₂, ...`) in one shot

No learning rate, no epochs, just one matrix equation.

## Project structure
```
linear-regression-normal-equation/
├── train.py            # model class + training/comparison script
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
python train.py
```

## Results
Running on the California housing dataset, the custom model matches scikit-learn almost exactly:

| Model | R² score |
|---|---|
| `linearRegressionNormal` (custom) | ~0.606 |
| `LinearRegression` (scikit-learn) | ~0.606 |

Both land on the same score, which makes sense — the normal equation has one unique closed-form solution, so any correct implementation converges to the same weights.

## What I learned
- Derived and implemented the normal equation `(XᵀX)⁻¹XᵀY` directly with NumPy
- Understood why a bias column of 1s needs to be added to `X` to solve for the intercept in the same matrix operation
- Validated correctness by comparing against scikit-learn's implementation rather than just trusting the math
- Saw firsthand why the normal equation has no hyperparameters (learning rate, epochs) — it's a direct solve, not an iterative one

## Author

**Amine Ghezal**
