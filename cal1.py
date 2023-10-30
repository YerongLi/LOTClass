from sklearn.metrics import precision_score, recall_score, f1_score

# Read the ground truth labels
with open('datasets/news/movies_train_labels.txt', 'r') as file:
    true_labels = [int(line.strip()) for line in file]

# Read the predicted labels
with open('datasets/movies/out.txt', 'r') as file:
    predicted_labels = [int(line.strip()) for line in file]
true_labels=true_labels[:100]
print(set(true_labels))
predicted_labels=predicted_labels[:100]
print(set(predicted_labels))
# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')

# Display the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
