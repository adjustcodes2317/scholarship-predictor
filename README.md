# Scholarship Eligibility Predictor

A machine learning project that predicts whether a student is eligible for a scholarship based on their CGPA, family income, and number of achievements — built using only numpy, no external ML libraries.

---

## The Problem

Scholarship selection in most colleges is done manually — someone goes through each application and checks eligibility one by one. This is slow and sometimes inconsistent. I wanted to build a model that can learn from past scholarship data and automatically predict whether a new student is eligible.

---

## What It Does

- Creates a dataset of 200 students — 100 eligible and 100 not eligible
- Normalizes the data so all features are on the same scale
- Splits data into 80% training and 20% testing
- Trains a Logistic Regression model from scratch using numpy
- Evaluates using accuracy, precision, recall, F1 score, and confusion matrix
- Predicts eligibility for a new student with a probability score

---

## Features Used

| Feature | What it means |
|---------|--------------|
| CGPA | Academic score out of 10 |
| Family Income | Annual family income in Rs |
| Achievements | Number of awards or certificates (scale 0-5) |

---

## How to Run

Only numpy needed:

```
pip install numpy
```

Run the program:

```
python scholarship_predictor.py
```

---

## Example Output

```
Student Details:
  CGPA         : 8.2
  Family Income: Rs 250,000
  Achievements : 3 (out of 5)

Eligibility Probability : 99.17%
Result : ELIGIBLE FOR SCHOLARSHIP
```

---

## Project Structure

```
scholarship-eligibility-predictor/
│
├── scholarship_predictor.py   # main program
├── README.md                  # this file
└── report/Project_Report.pdf  # full project report
```

---

## Library Used

- numpy — dataset creation, normalization, matrix operations, gradient descent


---

