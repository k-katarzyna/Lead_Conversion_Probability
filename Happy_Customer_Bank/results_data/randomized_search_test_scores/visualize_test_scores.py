# checking at what number of iterations the result stops improving

from joblib import load
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.spines.top": False})
plt.rcParams.update({"axes.spines.right": False})


test_scores = load("test_scores.pkl")
max_score_so_far = {}

for key, values in test_scores.items():
    max_scores = []
    current_max = values[0]
    
    for value in values:
        if value > current_max:
            current_max = value
        max_scores.append(current_max)
    max_score_so_far[key] = max_scores

    
plt.figure(figsize = (12, 6))
for model_name, scores in max_score_so_far.items():
    plt.plot(range(len(scores)), scores, label = model_name)
plt.legend()
plt.title("Number of randomized search iterations vs. maximum score achieved in subsequent iterations")  
plt.ylabel("Best ROC AUC achieved")
plt.xlabel("Iterations")
plt.grid(linewidth = 0.2, color = "gray", alpha = 0.5)
plt.savefig("iterations_vs_max_score.png", dpi = 150)