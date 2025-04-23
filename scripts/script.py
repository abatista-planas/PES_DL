import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_pickle("accuracy_training_size.pkl")
hyperparameter = "trainng_size"

fig,ax = plt.subplots(1,figsize=(12,6))
for column in df.drop("variation_list", axis=1).columns:
    accuracy_list = df[column].to_numpy()
    variation_list = df["variation_list"].to_numpy()
    ax.plot(variation_list,accuracy_list,'o-',label=column,markerfacecolor='w',markersize=9)

ax.set_ylabel('accuracy')
ax.set_xlabel(hyperparameter)
ax.set_title('Accuracy vs '+ hyperparameter)
if hyperparameter == "lr":
    ax.set_xscale('log')
ax.legend()

plt.show()