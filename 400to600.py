# Day 678 May 28 2023 Sunday Week 97 💯
#working on route project v4

# Day 679 May 29 2023 Tuesday Week 97 💯

import numpy as np
import pandas as pd

df = pd.DataFrame({'Column':['Normal 1', 'Normal 2', 'Normal 3']})
df

dfArray= df.values

newdf = np.insert(dfArray, obj = 1, values = ['Abnormal'], axis = 0)

df = pd.DataFrame(newdf, columns = ['Column'])
df


# Day 680 May 31 2023 Wednesday Week 97 💯


import numpy as np

import pandas as pd
  
df = pd.DataFrame({'Animal': ['Dog', 'Cat', 'Bird'],
                   'Type': ['Mammal', 'Mammal', 'Bird'],
                   'Color': ['Brown', 'Black', 'Yellow']})

df

df.values

dfArray = df.values

dfArray = np.insert(arr = dfArray, obj = 2, values = ['Bear', 'Mammal', 'Black'], axis = 0 )

df = pd.DataFrame(dfArray, columns = ['Animal', 'Type', 'Color'])
df

# Day 681 June 1 2023 Thursday Week 97 💯


import pandas as pd
  
df = pd.DataFrame({'Animal': ['Dog', 'Cat', 'Bird'],
                   'Type': ['Mammal', 'Mammal', 'Bird'],
                   'Color': ['Brown', 'Black', 'Yellow']})

df['Animal'].str.lower()

# Day 682 June 2 2023 Friday Week 97 💯


import pandas as pd
  
df = pd.DataFrame({'Animal': ['Dog', 'Cat', 'Bird'],
                   'Type': ['Mammal', 'Mammal', 'Bird'],
                   'Color': ['Brown', 'Black', 'Yellow']})

df['Colorf'] = df['Color'].str.upper()

df

# Day 683 June 3 2023 Saturday Week 97 💯


import pandas as pd
  
df = pd.DataFrame({'Animal': ['Dog', 'Cat', 'Bird'],
                   'Type': ['Mammal', 'Mammal', 'Bird'],
                   'Color': ['Brown', 'Black', 'Yellow']})

df['Color'].apply(lambda x: x + ' HEX')

# Day 684 June 4 2023 Sunday Week 97 💯


import pandas as pd
  
df = pd.DataFrame({'Animal': ['Dog', 'Cat', 'Bird'],
                   'Type': ['Mammal', 'Mammal', 'Bird'],
                   'Color': ['Brown', 'Black', 'Yellow']})

df[df['Animal'].str.contains("D")] #case sensitive


# Day 685 June 5 2023 Monday  💯


import pandas as pd

df_new = pd.DataFrame({'Name': ['John', 'Mary', 'Peter'],
                      'Gender': ['Male', 'Female', 'Male'],
                      'Age': [25, 35, 42]})


df_new['Name'] =df_new['Name'].apply(lambda x: x + " Roberts")

df_new


# Day 685 June 5 2023 Monday  💯


import pandas as pd

df_new = pd.DataFrame({'Name': ['John', 'Mary', 'Peter'],
                      'Gender': ['Male', 'Female', 'Male'],
                      'Age': [25, 35, 42]})


df_new['Name'] =df_new['Name'].apply(lambda x: x + " Roberts")

df_new

# Day 686 June 6 2023 Tuesday  💯


import pandas as pd

# First dataframe
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Jane', 'Mark', 'Lisa', 'Mike'],
    'Age': [25, 32, 20, 29, 35]
})

# Second dataframe
df2 = pd.DataFrame({
    'ID': [1, 3, 5, 6, 7],
    'Salary': [50000, 60000, 70000, 55000, 45000],
    'Department': ['Sales', 'Marketing', 'Finance', 'IT', 'HR']
})


df1

df2

mergedDf = df1.merge(df2, on = "ID")

mergedDf

# Day 687 June 7 2023 Wednesday  💯


import pandas as pd

# First dataframe
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Jane', 'Mark', 'Lisa', 'Mike'],
    'Age': [25, 32, 20, 29, 35]
})

# Second dataframe
df2 = pd.DataFrame({
    'ID': [1, 3, 5, 6, 7],
    'Salary': [50000, 60000, 70000, 55000, 45000],
    'Department': ['Sales', 'Marketing', 'Finance', 'IT', 'HR']
})

mergedDf = df1.merge(df2, on = "ID")
mergedDf 

mergedDf[mergedDf['Name'].str.contains("John")]

# Day 689 June 8 2023 Thursday  💯


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np

mean = 50
std = 2
x =  np.linspace(mean - 3*std, mean + 3*std, 1000) 
y =  norm.pdf(x, loc = mean, scale = std)
plt.plot(x,y)

# Day 689 June 9 2023 Friday  💯


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np

fig, graph1 = plt.subplots(figsize = (5,4))
graph1.set_facecolor("#fefae0")

k = 50
mean = 50
std = 2
x =  np.linspace(mean - 3*std, mean + 3*std, 1000) 
y =  norm.pdf(x, loc = mean, scale = std)
graph1.plot(x,y, color = "#ae2012")

x_fill=  np.linspace(mean - 3*std,k, 1000) 
y_fill= norm.pdf(x_fill, loc = mean, scale = std)

plt.fill_between(x_fill, y_fill, color = "#ee9b00")

graph1.set_title(f'Normal Distribution µ = {mean} σ = {std}' )

p = round(norm.cdf(x = k, loc = mean, scale = std), 4)
#p = print("{:.0%}".format(p))

plt.text(x = 48, y = 0.075, s = p, fontsize = 15)

# Day 690 June 10 2023 Saturday  💯


import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

fig, (graph1, graph2) = plt.subplots(figsize = (10,4), nrows = 1, ncols = 2)


#Graph 1

mean = 0
std = 1

x_values =  np.linspace(mean - 3*std, mean + 3*std, 1000)
y_values = norm.pdf(x_values)
graph1.plot(x_values, y_values)

x_fill = np.linspace(mean - 3*std, mean, 1000)
y_fill = norm.pdf(x_fill, loc = mean, scale = std)
graph1.fill_between(x_fill, y_fill, color = "#457b9d")
graph1.text(-1.1,0.15, "50%", color = "white", fontsize = 15)

graph1.set_facecolor("#f1faee")
graph1.set_title(f'Standard Distribution µ = {mean} σ = {std}')


#Graph 2
mean = 100
std = 10
x_values =  np.linspace(mean - 3*std, mean + 3*std, 1000)
y_values = norm.pdf(x_values, loc = mean, scale = std)
graph2.plot(x_values, y_values)

x_fill = np.linspace(mean - 3*std, mean, 1000)
y_fill = norm.pdf(x_fill, loc = mean, scale = std)
graph2.fill_between(x_fill, y_fill, color = "#457b9d")

fig.set_facecolor("#a8dadc")
graph2.set_facecolor("#f1faee")
graph2.set_title(f'Normal Distribution µ = {mean} σ = {std}')
graph2.text(90, 0.015, "50%", color = "white", fontsize = 15)

# Day 691 June 11 2023 Sunday  💯


import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

fig, (graph1, graph2) = plt.subplots(figsize = (10,4), nrows = 1, ncols = 2)


#Graph 1

mean = 0
std = 1

fig.set_facecolor("#eeeeee")


x_values =  np.linspace(mean - 3*std, mean + 3*std, 1000)
y_values = norm.pdf(x_values)
graph1.plot(x_values, y_values, color = "black")

x_fill = np.linspace(mean - 3*std, mean, 1000)
y_fill = norm.pdf(x_fill, loc = mean, scale = std)
graph1.fill_between(x_fill, y_fill, color = "white")
graph1.text(-1.1,0.15, "50%", color = "black", fontsize = 15)

graph1.set_facecolor("#eeeeee")
graph1.set_title(f'Standard Normal Distribution µ = {mean} σ = {std}')
graph1.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
graph1.set_yticks([])


#Graph 2
mean = 100
std = 10
x_values =  np.linspace(mean - 3*std, mean + 3*std, 1000)
y_values = norm.pdf(x_values, loc = mean, scale = std)
graph2.plot(x_values, y_values, color = "black")

x_fill = np.linspace(mean - 3*std, mean, 1000)
y_fill = norm.pdf(x_fill, loc = mean, scale = std)
graph2.fill_between(x_fill, y_fill, color = "white")

graph2.set_facecolor("#eeeeee")
graph2.set_title(f'Normal Distribution µ = {mean} σ = {std}')
graph2.text(90, 0.015, "50%", color = "black", fontsize = 15)
graph2.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
graph2.set_yticks([])

# Day 692 June 12 2023 Monday  💯


import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

fig, (graph1, graph2) = plt.subplots(figsize = (10,4), nrows = 1, ncols = 2)


#Graph 1

mean = 0
std = 1

fig.set_facecolor("#eeeeee")

k = 3
norm.cdf(k , loc = mean, scale = std)

x_values =  np.linspace(mean - 3*std, mean + 3*std, 1000)
y_values = norm.pdf(x_values)
graph1.plot(x_values, y_values, color = "black")

x_fill = np.linspace(mean - 3*std, k, 1000)
y_fill = norm.pdf(x_fill, loc = mean, scale = std)
graph1.fill_between(x_fill, y_fill, color = "black")
graph1.text(-0.5,0.15, "100%", color = "white", fontsize = 15)

graph1.set_facecolor("#eeeeee")
graph1.set_title(f'Standard Normal Distribution µ = {mean} σ = {std}')
graph1.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
graph1.set_yticks([])

#Graph 2
k = 110
norm.cdf(k , loc = mean, scale = std)
mean = 100
std = 10
x_values =  np.linspace(mean - 3*std, mean + 3*std, 1000)
y_values = norm.pdf(x_values, loc = mean, scale = std)
graph2.plot(x_values, y_values, color = "black")

x_fill = np.linspace(mean - 3*std, k, 1000)
y_fill = norm.pdf(x_fill, loc = mean, scale = std)
graph2.fill_between(x_fill, y_fill, color = "black")

graph2.set_facecolor("#eeeeee")
graph2.set_title(f'Normal Distribution µ = {mean} σ = {std}')
graph2.text(95, 0.015, "84%", color = "white", fontsize = 15)
graph2.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
graph2.set_yticks([])

# Day 690 June 13 2023 Tuesday  💯


import numpy as np

x = np.arange(start = 0, stop = 6.5, step =  1)
x

x = np.linspace(start = -3, stop = 3, num = 5)
x

# Day 691 June 14 2023 Wednesday  💯


import numpy as np
np.add([5, 34, 2], [4, 6, 7])







