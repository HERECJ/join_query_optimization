import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name2 = "Tree_DQN_1026-142734"
f = open('logs/{}/running_log.txt'.format(name2), "r") #
lines = f.readlines()  #
plot_x=[]
plot_loss=[]
name=""
for num in range(len(lines)):

    a=1
    if a==0:
        if "Loss" in lines[num]:
            episode = int(lines[num].split("Episode")[1].strip().split(",")[0].strip())
            loss = float(lines[num].split("Loss :")[1].strip())

            plot_x.append(episode)
            plot_loss.append(loss)
            name = "Loss"
    else:
        if "Avg Cost" in lines[num]:
            episode = int(lines[num].split("/")[1].strip().split(",")[0].strip())
            loss = float(lines[num].split("Avg Cost")[1].strip())

            plot_x.append(episode)
            plot_loss.append(loss)
            name = "Avg Cost"






x = np.array(plot_x[1:])
y = np.array(plot_loss[1:])
print(np.argmin(plot_loss[1:]))
print(np.min(plot_loss[1:]))
print(x[np.argmin(plot_loss[1:])])
df = pd.DataFrame({"step": x, name: y})
sns.scatterplot(x='step',y=name,data=df)
#print(df)
sns.lineplot(x=x, y=y)
plt.title(name2)
#plt.ylim(0,100000000000000)
# plt.show()
plt.savefig('view.jpg')