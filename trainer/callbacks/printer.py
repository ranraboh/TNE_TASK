from transformers import Trainer
import torch
from transformers import TrainerCallback
import matplotlib.pyplot as plt


# Callbacks
class PrinterCallback(TrainerCallback):
    """
        DESCRIPTION: After the model completes the training process,
        The printer callback display the evaluation accuracy of the model over
        the epochs by plotting a graph.
        Used for diagnostics and analysis.
    """
    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    # Print a graph showing the progress of the model.
    # the graph display the evaluation accuracy of the model for
    # for each epoch along the training process.
    def on_train_end(self, args, state, control, **kwargs):
        log_data = state.log_history

        # Assign y axis values (epochs eval accuracy)
        y = []
        for i in range(len(log_data)):
          print (log_data[i])
          if "eval_accuracy" in log_data[i]:
            y.append(100 * float(log_data[i]["eval_accuracy"]))

        # Assign x axis values (epochs range)
        x = torch.arange(len(y))

        # Naming x and y axis
        plt.xlabel('Epoch')
        plt.ylabel('Eval accuracy')
        plt.plot(x, y)

        # Provide title for the plot
        plt.title('Summary')

        # Display the plot
        plt.show()
