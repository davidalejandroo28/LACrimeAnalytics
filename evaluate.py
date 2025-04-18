import os
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, 
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt

def evaluate_model(eval_y, pred_y, model_name, best_params, save_path):
    # Ensure the save_path directory exists.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Read the crime mapping CSV to determine display labels.
    mapping_df = pd.read_csv("./data/crime_mapping.csv")
    # Sort by crime code in case they are not ordered and get the list of labels.
    display_labels = mapping_df.sort_values("crime_code")["crime_code"].tolist()
    
    # Compute confusion matrix.
    cm = confusion_matrix(eval_y, pred_y)
    
    # Create and plot the confusion matrix with the labels from the CSV.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
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
