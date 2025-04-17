import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix  # type: ignore


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plots the confusion matrix for the given true and predicted labels."""

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print the confusion matrix
    print("Confusion Matrix: ", title)
    print(cm)

    # Visualize the confusion matrix
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add text annotations to the plot
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.show()


def sample_visualization(
    df_samples: pd.DataFrame,
    nrow: int = 0,
    ncol: int = 0,
    index_array: npt.NDArray[np.int8] = np.array([]),
):
    """Visualizes the samples generated by the DataGenerator"""
  
    
   
    if index_array.size == 0:
        index_array = np.arange(len(df_samples))
        df_samples.reset_index(drop=True, inplace=True)
        
    df_plot = df_samples[df_samples.index.isin(index_array)]    
    
    if nrow == 0 or ncol == 0:
        df_size = min(len(df_plot),15)
        if df_size <= 3:
            nrow=df_size
            ncol=1
        elif df_size <= 8 or df_size ==10:
            nrow=int((df_size+1)/2)
            ncol=2    
        else:
            nrow=int((df_size+2)/3)
            ncol=3 
            
    fig, axs = plt.subplots(nrow, ncol)
    fig.set_size_inches(20, 20)        
    count = 0
    def custom_plot(row):
        """Custom plot function to plot each row of the DataFrame."""
        
        nonlocal count
     
        if count < 15:
            row.pes.plot(ax=axs[int(count / ncol)][count % ncol],label= row.deformation_type, x="r", y="energy")     
        count += 1

    
      
    df_plot.apply(lambda x: custom_plot(x), axis=1)
    

    plt.show()
