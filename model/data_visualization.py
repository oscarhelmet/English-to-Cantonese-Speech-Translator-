import matplotlib.pyplot as plt
import numpy as np

def plot_image(losses, file_name="data_visualization.png"):
    window_size = 50
    smoothed_losses = []
    for i in range(len(losses)-window_size):
        smoothed_losses.append(np.mean(losses[i:i+window_size]))
    
    plt.figure(figsize=(10, 5))  
    plt.plot(smoothed_losses[100:])
    plt.title("Smoothed Losses Over Time")  
    plt.xlabel("Epochs")  
    plt.ylabel("Loss")  
    plt.savefig(file_name)  
    plt.close()  