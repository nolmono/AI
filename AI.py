import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = {'task': [2000,2001,2002,2003,1000,2599,2456,2333,4433,2222,6777,867,23454,5665,444556,6544],
        'done': [1000,1394,1377,1203,999,2500,2430,2333,4430,2111,6333,866, 21123,5432,345666,6433]        
                }
print(data)

df=pd.DataFrame(data,columns=['task','done'])

task = pd.DataFrame(data["task"])
done = pd.DataFrame(data["done"])
print(task)
linear_model = linear_model.LinearRegression()
model = linear_model.fit(task, done)
print(model.coef_)
print(model.intercept_)
print("The Equation: y = "+str(model.coef_[0][0]) + "and x: " + str(model.intercept_[0]) )

list_task = [int(row) for row in task.values]
list_done = [int(row) for row in done.values]

n = len(task)
sum = 0
for i in range (0, n):
    diff = list_task[i] - list_done[i]
    sq_diff = diff**2
    sum = sum + sq_diff
MSE = sum/n
print("The MSE  = " + str(MSE))

plt.scatter(df['task'], df['done'], color='red')
plt.plot(task, done)
plt.xlabel("task")
plt.ylabel("done")
plt.grid(True)
plt.show()

#data.plot(kind="scatter", x="task", y= "done")
plt.plot(task, model.predict(task))
plt.show()




