from ray.tune import ExperimentAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Load experiment analysis
df = pd.read_pickle('experiment_results.pkl')

print(df['config/hidden_layers'])

# Plot loss vs learning rate
sns.scatterplot(x="config/lr", y="loss", data=df)
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate vs Loss")
plt.show()
