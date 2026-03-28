import numpy as np

print("=" * 50)
print("   SCHOLARSHIP ELIGIBILITY PREDICTOR")
print("=" * 50)

np.random.seed(42)

cgpa1    = np.random.normal(8.5, 0.5,  100)
income1  = np.random.normal(200000, 50000, 100)
achieve1 = np.random.normal(3, 1, 100)

cgpa2    = np.random.normal(6.5, 0.7,  100)
income2  = np.random.normal(600000, 80000, 100)
achieve2 = np.random.normal(1, 0.5, 100)

# clip to realistic ranges
cgpa1    = np.clip(cgpa1,    6.0, 10.0)
cgpa2    = np.clip(cgpa2,    4.0,  9.0)
income1  = np.clip(income1,  50000,  400000)
income2  = np.clip(income2,  300000, 900000)
achieve1 = np.clip(achieve1, 1, 5)
achieve2 = np.clip(achieve2, 0, 3)

# combine into one dataset
cgpa    = np.concatenate([cgpa1, cgpa2])
income  = np.concatenate([income1, income2])
achieve = np.concatenate([achieve1, achieve2])

y = np.array([1]*100 + [0]*100)
X = np.column_stack([cgpa, income, achieve])

print(f"\nDataset Created  : 200 Students")
print(f"Eligible (1)     : {int(np.sum(y == 1))}")
print(f"Not Eligible (0) : {int(np.sum(y == 0))}")

mean = np.mean(X, axis=0)
std  = np.std(X,  axis=0)
X    = (X - mean) / std

print("Data Normalized")


split   = int(0.8 * len(y))
indices = np.arange(len(y))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training Samples : {len(y_train)}")
print(f"Testing Samples  : {len(y_test)}")


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0
    for i in range(epochs):
        z     = np.dot(X, w) + b
        pred  = sigmoid(z)
        error = pred - y
        w -= lr * np.dot(X.T, error) / len(y)
        b -= lr * np.mean(error)
    return w, b

def predict(X, w, b):
    prob = sigmoid(np.dot(X, w) + b)
    return (prob >= 0.5).astype(int), prob

w, b = train(X_train, y_train)
print("\nModel Trained Successfully!")


y_pred, y_prob = predict(X_test, w, b)

accuracy  = np.mean(y_pred == y_test) * 100
tp = int(np.sum((y_pred == 1) & (y_test == 1)))
tn = int(np.sum((y_pred == 0) & (y_test == 0)))
fp = int(np.sum((y_pred == 1) & (y_test == 0)))
fn = int(np.sum((y_pred == 0) & (y_test == 1)))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0

print(f"\nConfusion Matrix:")
print(f"  True  Positive : {tp}")
print(f"  True  Negative : {tn}")
print(f"  False Positive : {fp}")
print(f"  False Negative : {fn}")


print("\n--- NEW STUDENT PREDICTION ---")

new_cgpa    = 8.2
new_income  = 250000
new_achieve = 3    # achievements out of 5

new       = np.array([[new_cgpa, new_income, new_achieve]])
new       = (new - mean) / std
pred, prob = predict(new, w, b)

print(f"\nStudent Details:")
print(f"  CGPA         : {new_cgpa}")
print(f"  Family Income: Rs {new_income:,}")
print(f"  Achievements : {new_achieve} (out of 5)")
print(f"\nEligibility Probability : {prob[0]*100:.2f}%")

if pred[0] == 1:
    print("Result : ELIGIBLE FOR SCHOLARSHIP")
else:
    print("Result : NOT ELIGIBLE")

print("\n" + "=" * 50)
print("   PROJECT COMPLETE")
print("=" * 50)
print(f"   Algorithm : Logistic Regression ")
print(f"   Dataset   : 200 students, 3 features")
print(f"   Library   : numpy only")
print(f"   Accuracy  : {accuracy:.2f}%")

print("=" * 50)
