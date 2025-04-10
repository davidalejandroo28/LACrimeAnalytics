import os
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, 
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt

def evaluate_model(eval_y, pred_y, model_name, best_params, save_path):
    """
    Evaluates model predictions, saves a confusion matrix plot and writes overall statistics.
    
    Parameters:
        eval_y (array-like): True labels.
        pred_y (array-like): Predicted labels.
        model_name (str): A name for the model used in the plot title and file names.
        best_params (dict): The best hyperparameters obtained via GridSearchCV.
        save_path (str): Directory in which the output files will be saved.
    
    Saves:
        - A confusion matrix plot as '<model_name>_confusion_matrix.png' in save_path.
        - A text file '<model_name>_performance_report.txt' in save_path that includes overall accuracy,
          precision, recall, F1 score, and the best hyperparameters.
    """
    # Ensure the save_path directory exists.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Compute confusion matrix with labels 0 to 9.
    cm = confusion_matrix(eval_y, pred_y)
    display_labels = list(range(10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    # Plot the confusion matrix and save to file.
    disp.plot(cmap=plt.cm.Reds)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plot_filename = os.path.join(save_path, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to avoid display
    
    # Calculate overall metrics using weighted averaging.
    accuracy = accuracy_score(eval_y, pred_y)
    precision, recall, f1, _ = precision_recall_fscore_support(eval_y, pred_y, average='weighted')
    
    # Prepare a report string.
    report = (
        f"Model: {model_name}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision (weighted): {precision:.4f}\n"
        f"Recall (weighted): {recall:.4f}\n"
        f"F1 Score (weighted): {f1:.4f}\n\n"
        f"Best Hyperparameters:\n{best_params}\n"
    )
    
    # Save the report to a text file.
    report_filename = os.path.join(save_path, f"{model_name}_performance_report.txt")
    with open(report_filename, "w") as f:
        f.write(report)
    
    print(report)
    print(f"Confusion matrix plot saved to: {plot_filename}")
    print(f"Performance report saved to: {report_filename}")
    
    return f1
