import matplotlib.pyplot as plt
import json
import os
import argparse

def plot_accuracy_curves(history_files):
    """
    Plots accuracy learning curves from history JSON files
    
    Args:
        history_files: List of history JSON file paths to plot
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    
    for i, history_path in enumerate(history_files):
        model_name = os.path.basename(history_path).replace('_history.json', '')
        color = colors[i % len(colors)]
        
        # Load training history
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            # Plot training and validation accuracy
            plt.plot(history["epochs"], history["train_acc"], f'{color}-', label=f'{model_name} - Train Acc')
            plt.plot(history["epochs"], history["val_acc"], f'{color}--', label=f'{model_name} - Val Acc')
            
            # Find and mark peak validation accuracy
            best_epoch = history["val_acc"].index(max(history["val_acc"])) 
            best_acc = max(history["val_acc"])
            plt.plot(history["epochs"][best_epoch], best_acc, f'{color}o', markersize=8)
            plt.annotate(f'{best_acc:.4f}', 
                        (history["epochs"][best_epoch], best_acc),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
            
        except Exception as e:
            print(f"Error loading or plotting history from {history_path}: {e}")
    
    # Set titles and labels
    plt.title('Model Accuracy During Training', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Add some margin to y-axis for annotations
    plt.ylim(top=plt.ylim()[1] * 1.05)
    
    plt.tight_layout()
    plt.savefig('accuracy_curves.png', dpi=300)
    plt.show()
    
    print(f"Accuracy curves saved to 'accuracy_curves.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot accuracy curves from model history files")
    parser.add_argument(
        "--history_files", 
        nargs='+',
        default=["checkpoints/early_stop_model_history.json", "checkpoints/final_model_history.json"],
        help="List of history JSON files to analyze"
    )
    
    args = parser.parse_args()
    
    print("Plotting accuracy curves for models:")
    for history_file in args.history_files:
        print(f" - {history_file}")
    
    plot_accuracy_curves(args.history_files) 