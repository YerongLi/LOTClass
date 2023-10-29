from sklearn.metrics import precision_score, recall_score, f1_score

# Read the ground truth labels
with open('datasets/movies/movies_train_labels.txt', 'r') as file:
    true_labels = [int(line.strip()) for line in file]

# Read the predicted labels
with open('datasets/movies/out.txt', 'r') as file:
    predicted_labels = [int(line.strip()) for line in file]
print(set(true_labels))
print(set(predicted_labels))
# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels, average='binary')
recall = recall_score(true_labels, predicted_labels, average='binary')
f1 = f1_score(true_labels, predicted_labels, average='binary')

# Display the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
