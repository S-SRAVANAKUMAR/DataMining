import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset and reduce size
ratings = pd.read_csv('ratings_small.csv')
ratings = ratings.head(10000)  # Use only the first 10,000 rows for faster computation

# Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_item_matrix = user_item_matrix.fillna(0)
R = user_item_matrix.to_numpy()

# Metrics functions
def evaluate(true_ratings, predicted_ratings):
    mask = true_ratings > 0  # Evaluate only on non-zero ratings
    mae = mean_absolute_error(true_ratings[mask], predicted_ratings[mask])
    rmse = np.sqrt(mean_squared_error(true_ratings[mask], predicted_ratings[mask]))
    return mae, rmse

# Matrix Factorization with reduced iterations
def matrix_factorization(R, K, steps=100, alpha=0.002, beta=0.02):
    num_users, num_items = R.shape
    P = np.random.rand(num_users, K)
    Q = np.random.rand(num_items, K)
    
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] > 0:
                    eij = R[i, j] - np.dot(P[i, :], Q[j, :])
                    P[i, :] += alpha * (2 * eij * Q[j, :] - beta * P[i, :])
                    Q[j, :] += alpha * (2 * eij * P[i, :] - beta * Q[j, :])
    return P, Q

# User-based Collaborative Filtering
def user_based_cf(R, user_id, k=10):
    similarity = cosine_similarity(R)
    similar_users = np.argsort(-similarity[user_id])[:k+1]
    similar_users = [u for u in similar_users if u != user_id]
    R_pred = np.zeros(R.shape[1])
    for item in range(R.shape[1]):
        num = sum(similarity[user_id, u] * R[u, item] for u in similar_users if R[u, item] > 0)
        den = sum(abs(similarity[user_id, u]) for u in similar_users if R[u, item] > 0)
        R_pred[item] = num / den if den != 0 else 0
    return R_pred

# Item-based Collaborative Filtering
def item_based_cf(R, user_id, k=10):
    similarity = cosine_similarity(R.T)
    R_pred = np.zeros(R.shape[1])
    for item in range(R.shape[1]):
        num = sum(similarity[item, j] * R[user_id, j] for j in range(R.shape[1]) if R[user_id, j] > 0)
        den = sum(abs(similarity[item, j]) for j in range(R.shape[1]) if R[user_id, j] > 0)
        R_pred[item] = num / den if den != 0 else 0
    return R_pred

# Evaluate Models
P, Q = matrix_factorization(R, K=10, steps=50)  # PMF
R_pred_pmf = np.dot(P, Q.T)  # Correct multiplication

user_id = 0  # Select a specific user for evaluation
pred_user_cf = user_based_cf(R, user_id, k=10)  # User CF
pred_item_cf = item_based_cf(R, user_id, k=10)  # Item CF

true_ratings = R[user_id]
mae_pmf, rmse_pmf = evaluate(true_ratings, R_pred_pmf[user_id])
mae_user, rmse_user = evaluate(true_ratings, pred_user_cf)
mae_item, rmse_item = evaluate(true_ratings, pred_item_cf)

print(f"PMF - MAE: {mae_pmf}, RMSE: {rmse_pmf}")
print(f"User-based CF - MAE: {mae_user}, RMSE: {rmse_user}")
print(f"Item-based CF - MAE: {mae_item}, RMSE: {rmse_item}")

# Impact of Neighbors
k_values = [5, 10, 20]
rmse_neighbors = {'UserCF': [], 'ItemCF': []}

for k in k_values:
    pred_user_cf = user_based_cf(R, user_id, k=k)
    pred_item_cf = item_based_cf(R, user_id, k=k)
    _, rmse_user = evaluate(true_ratings, pred_user_cf)
    _, rmse_item = evaluate(true_ratings, pred_item_cf)
    rmse_neighbors['UserCF'].append(rmse_user)
    rmse_neighbors['ItemCF'].append(rmse_item)

# Plot results
plt.plot(k_values, rmse_neighbors['UserCF'], label='UserCF')
plt.plot(k_values, rmse_neighbors['ItemCF'], label='ItemCF')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('RMSE')
plt.legend()
plt.title('Impact of Number of Neighbors on RMSE')
plt.show()
