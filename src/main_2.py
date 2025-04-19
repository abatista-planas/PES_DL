import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd


from sklearn.model_selection import train_test_split

# X = torch.linspace(1, 50, 50).reshape(-1, 1)

# torch.manual_seed(71)
# e = torch.randint(-8,9,(50,1), dtype=torch.float)
# y = 2*X +1 + e 

# # plt.scatter(X.numpy(), y.numpy())
# #plt.show()

# torch.manual_seed(59)

# class Model(nn.Module):
#     def __init__(self,in_features, out_features):
#         super(Model, self).__init__()
#         self.linear = nn.Linear(in_features, out_features)

#     def forward(self, x):
#         return self.linear(x)
    
# model = Model(1, 1)
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# losses = []

# for epoch in range(50):
#     model.train()
#     optimizer.zero_grad()
#     y_pred = model.forward(X)
#     loss = criterion(y_pred, y)
#     losses.append(loss.item())
#     print(f"Epoch {epoch+1}, Loss: {loss.item()}")
#     loss.backward()
#     optimizer.step()
    

# plt.plot(range(50), np.array(losses), color='blue')
# plt.show()

df = pd.read_csv("/home/albplanas/Desktop/Programming/Data/iris.csv")

    
# fig, axes = plt.subplots(nrows= 2, ncols= 2, figsize=(10, 7))
# fig.tight_layout()

# plots = [(0, 1), (2, 3), (0, 2), (1, 3)]
# colors = ['b','r', 'g']
# labels = ['Iris setosa', 'Iris virginica','Iris versicolor']

# for i, ax in enumerate(axes.flat):
#     for j in range(3):
#         x = df.columns[plots[i][0]]
#         y = df.columns[plots[i][1]]
#         ax.scatter(df[df['target'] == j][x], df[df['target'] == j][y], color=colors[j], label=labels[j])
#         ax.set(xlabel = x,ylabel = y)
        
# fig.legend(labels=labels, loc=3, bbox_to_anchor=(1, 0.85))
# plt.show()        



features = df.drop('target', axis=1).values
label = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=33)



X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)



y_train = torch.LongTensor(y_train)#.reshape(-1,1)
y_test = torch.LongTensor(y_test)#.reshape(-1,1)



# data = df.drop('target', axis=1).values
# labels = df['target'].values

# iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
# iris_loader = DataLoader(iris, batch_size=15, shuffle=True)



class Model(nn.Module):
    def __init__(self,in_features = 4, h1 = 8,h2 = 9,out_features = 3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        pass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

torch.manual_seed(32)

model =Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for epoch in range(epochs):
    
    y_pred = model.forward(X_train)
    model.train()
    
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())
    if epoch % 10==0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# plt.plot(range(epochs), np.array(losses), color='blue')
# plt.show() 

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(f"Test Loss: {loss.item()}")
    print("Predictions: ", torch.argmax(y_eval, dim=1))
    print("Predictions: ",y_test)  
    print("Predictions: ",torch.argmax(y_eval, dim=1) == y_test)
    print("Accuracy (%): ", 100*torch.sum(torch.argmax(y_eval, dim=1) == y_test).item() / len(y_test))