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

# Day 692 June 15 2023 Thursday  💯


import numpy as np

np.subtract([10, 9, 8, 7], [1, 2, 3, 4])

# Day 693 June 16 2023 Friday  💯
import numpy as np
np.random.randint(low = 1, high = 10, size = 5)

# Day 694 June 17 2023 Saturday  💯

import numpy as np 
np.random.randint(low = 100, high = 105, size = 2)

# Day 695 June 18 2023 Sunday  💯

import numpy as np
np.random.randint(low = 10, high =  50 , size = 50)

# Day 696 June 19 2023 Monday  💯


import numpy as np
np.random.seed(0)

np.random.randint(low = -10 , high = 0, size = 3 ) 

# Day 697 June 20 2023 Tuesday  💯

import numpy as np
np.random.seed(0)
np.random.randint(low = 100, high = 1000, size = 100)

# Day 698 June 21 2023 Wednesday  💯
Libraries can contain modules and modules functions

import numpy as np
choices = [4, 56, 3, 24, 2, 4]
np.random.seed(100)
np.random.choice(choices)

# Day 699 June 22 2023 Thursday  💯
import numpy as np
choices = ['Sarai', 'Gen', 'Lizzy', 'Mom', 'Dad', 'Dog']
np.random.choice(choices)

# Day 700 June 23 2023 Friday  💯

import numpy as np
np.random.randint(low = 1000, high = 1500 , size = 6 )

# Day 702 June 24 2023 Saturday 💯

import numpy as np
np.random.randint(low = 20, high = 50, size = 3) 

# Day 702 June 25 2023 Sunday  💯


import numpy as np
choices = ['Melissa', 'Rojas', 'Maxwell', '5']
np.random.seed(100)
np.random.choice(choices)

# Day 703 June 26 2023 Monday  💯

rand = 1 row 2 columns -> generate random data

import numpy as np
import pandas as pd
np.random.seed(0)
data = np.random.rand(4,2)
data

df = pd.DataFrame(data, columns = ["Column", 'Column B'])
df

# Day 704 June 27 2023 Tuesday  💯
import numpy as np
np.random.randint(low = 2, high = 5, size = 1)


# Day 705 June 28 2023 Wednesday 💯
import numpy as np
np.random.sample([1, 2])


# Day 706 June 29 2023 Thursday 💯
import numpy as np
np.random.rand(1, 2)


# Day 707 June 30 2023 Friday 💯
import numpy as np
np.random.rand(5, 1)

np.random.sample([5, 1])

# Day 708 July 1 2023 Saturday 💯

import numpy as np
np.random.randint(low = 2, high = 10, size = 4)


# Day 709 July 2 2023 Sunday 💯

import numpy as np
np.random.rand(3, 4)


# Day 710 July 3 2023 Monday 💯
import numpy as np
np.random.sample([3, 4])


# Day 711 July 4 2023 Tuesday 💯

Create a 2D NumPy array of shape (3, 4) containing random numbers between 0 and 1

import numpy as np
np.random.rand(3,4)

# Day 712 July 5 2023 Wednesday 💯

import numpy as np

ids = ['Chelsea', 'Kylie', 'Felicia', 'Kayla', 'Mike', 'Ariel', 'Kendel']

np.random.choice(ids)

# Day 713 July 6 2023 Thursday 💯


Suppose you are a youtuber and are having a giveaway for your merch. You have the emails of the top 4 finalists and you can only choose 2. What you gonna do??

import numpy as np

customer_emails = ['customer1@example.com', 'customer2@example.com', 'customer3@example.com',  'customer1000@example.com']

np.random.choice(customer_emails, size = 2, replace = False)

Replace = False tells numpy that it cant RE-Place the email back into the pool of choices once its chosen
# Day 713 July 6 2023 Thursday 💯


import numpy as np
np.random.rand(2,4)


# Day 714 July 8 2023 Saturday 💯

https://numpy.org/doc/1.16/reference/routines.random.html

import numpy as np
np.random.sample(size = [2, 3])

# Day 715 July 9 2023 Sunday 💯


import numpy as np
np.random.rand(2,3)
# Day 716 July 10 2023 Monday 💯


import numpy as np
choices = [4, 35, 3, 24, 3]
np.random.choice(choices, size = 3, replace = False)

# Day 717 July 11 2023 Tuesday 💯


import numpy as np
np.random.rand(10,10)

# Day 718 July 12 2023 Wednesday 💯
import numpy as np
np.random.sample([10,10])


# Day 719 July 13 2023 Thursday 💯
import numpy as np
np.random.rand(4, 4)


# Day 720 July 14 2023 Friday 💯
import numpy as np
np.random.sample([2,3]);

# Day 721 July 15 2023 Friday 💯

choices = ['Personality1', 'Personality 2']

choice = np.random.choice(choices, size = 1, replace = False) 
choice[0]

# Day 722 July 16 2023 Saturday 💯
import numpy as np
np.random.sample([2,2])
# Day 723 July 17 2023 Sunday 💯

import numpy as np
np.random.rand(1, 5)
# Day 724 July 18 2023 Monday 💯

import numpy as np

arr = ['a', 'b', 'c', 'd']

np.random.shuffle(arr)

arr

# Day 725 July 19 2023 Tuesday 💯

import numpy as np
np.linspace(-3, 3, 1000)

# Day 726 July 20 2023 Thursday 💯

import numpy as np
np.random.rand(1,2)


# Day 727 July 21 2023 Friday 💯

import numpy as np
np.random.sample([1,5])


# Day 728 July 22 2023 Saturday 💯


https://www.khanacademy.org/math/ap-statistics/density-curves-normal-distribution-ap/normal-distributions-calculations/e/z_scores_2

![image.png](data:image/png;base64,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)

from scipy.stats import norm
prop = round(norm.cdf(x = 90.4, loc = mean, scale = std),4)
prop

import matplotlib.pyplot as plt
import numpy as np
value = 90.4
mean =  105
std = 10
x =  np.linspace(start =  mean - (3* std) , stop = mean + (3*std) , num = 1000)
y = norm.pdf(x, loc = mean , scale = std)
plt.plot(x,y, color = "black")
plt.title(f'Normal Distribution Given µ = {mean} σ = {std}')

x_fill = np.linspace(start = mean - (3*std), stop = value, num = 100)
y_fill = norm.pdf(x_fill, loc = mean , scale = std)


plt.fill_between(x_fill, y_fill, color = "#48cae4")
plt.text(x = 83, y = 0.002, s = f'{prop}', color = "white", fontstyle = "oblique")

ahh isnt she cute

![image.png](data:image/png;base64,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REREREREZGE4hg4RERERERERkYVhQIeIiIiIiIiIyMIwoENEREREREREZGEY0CEiIiIiIiIisjAM6BARERERERERWRgGdIiIiIiIiIiILAwDOkREREREREREFoYBHSIiIiIiIiIiC8OADhERERERERGRhWFAh4iIiIiIiIjIwjCgQ0RERERERERkYRjQISIiIiIiIiKyMAzoEBERERERERFZGAZ0iIiIiIiIiIgsDAM6REREREREREQWhgEdIiIiIiIiIiILw4AOEREREREREZGFYUCHiIiIiIiIiMjCMKBDRERERERERGRhGNAhIiIiIiIiIrIwDOgQEREREREREVkYBnSIiIiIiIiIiCwMAzpERERERERERBaGAR0iIiIiIiIiIgvDgA4RERERERERkYVhQIeIiIiIiIiIyMIwoENEREREREREZGEY0CEiIiIiIiIisjAM6BARERERERERWRgGdIiIiIiIiIiILAwDOkREREREREREFoYBHSIiIiIiIiIiC8OADhERERERERGRhWFAh4iIiIiIiIjIwjCgQ0RERERERERkYRjQISIiIiIiIiKyMAzoEBERERERERFZGAZ0iIiIiIiIiIgsDAM6REREREREREQWhgEdIiIiIiIiIiILw4AOEREREREREZGFsTofdkFR/yYiIiIiIiKiB1yAv5/6F1kyK0VQ/yYiIiIiIiIiIgvAW66IiIiIiIiIiCwMAzpERERERERERBaGAR0iIiIiIiIiIgvDgA4RERERERERkYVhQIeIiIiIiIiIyMIwoENEREREREREZGEY0CEiIiIiIiIisjAM6BARERERERERWRgGdIiIiIiIiIiILAwDOkREREREREREFoYBHSIiIiIiIiIiC8OADhERERERERGRhWFAh4iIiIiIiIjIwjCgQ0RERERERERkYRjQISIiIiIiIiKyMAzoEBERERERERFZGAZ0iIiIiIiIiIgsDAM6REREREREREQWhgEdIiIiIiIiIiILw4AOEREREREREZGFYUCHiIiIiIiIiMjCMKBDRERERERERGRhGNAhIiIiIiIiIrIwDOgQEREREREREVkYBnSIiIiIiIiIiCwMAzpERERERERERBaGAR0iIiIiIiIiIgvDgA4RERERERERkYVhQIeIiIiIiIiIyMJYKYL6NxEREd0j58+fx/bt2xEVFaXO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Tpw4oZsnT/DlrTVyio7WP1pJvieXkcvKz5SECpXECa+Y8j3RLieDHsUM5kjy5Lqgs2v1u4oSzJEq2Mn9MfMMXgaoClhWt74i7nd62G4cdO+F1mYcOjIA06BBg2JPzZs3L5NgjqQrS6aSU+Z7YfNUVw6KF4nJc3uIiIgeUEXsPhERkeVyQaPOQbA9sgI/fL8Sm3cfxP4DctqnG4Nm+m8h6DpmDJrlGNiUSpa8fUo+ntqYqYtm5TJyWbrfJeN0VFUMecwbjCkQERFRaeAtV0RED6ubcTh17hwiz8VBf+OHI2rWqYW6dQJRI8dQOVR8xrdcGZ5eZCAHQjZ3XnFvuSIiIiKiBwMDOkRERKXAOKBTHAzoEBEREZHEgA4REVEpmDx5Mo4cOYKRI0eqc4rmp59+0o2bMm3aNHUOERERET2MGNAhIiIqBfJR03KQ49DQUHVO0chHmjdq1AgNGzZU5xARERHRw4gBHSIiIiIiIiIiC8OnXBERERERERERWRgGdIiIiIiIiIiILAwDOkREREREREREFoYBHSIiIiIiIiIiC8OADhERERERERGRhWFAh4iIiIiIiIjIwjCgQ0RERERERERkYRjQISIiIiIiIiKyMAzoEBERERERERFZGAZ0iIiIiIiIiIgsDAM6REREREREREQWhgEdIiIiIiIiIiILw4AOEREREREREZGFYUCHiIiIiIiIiMjCMKBDRERERERERGRhGNAhIiIiIiIiIrIwDOgQEREREREREVkYBnSIiIiIiIiIiCwMAzpERERERERERBaGAR0iIiIiIiIiIgvDgA4RERERERERkYVhQIeIiIiIiIiIyMIwoENEREREREREZGEY0CEiIiIiIiIisjAM6BARERERERERWRgGdIiIiIiIiIiILAwDOkREREREREREFoYBHSIiIiIiIiIiC8OADhERERERERGRhWFAh4iIiIiIiIjIwjCgQ0RERERERERkYRjQISIiIiIiIiKyMAzoEBERERERERFZGAZ0iIiIiIiIiIgsDAM6REREREREREQWhgEdIiIiIiIiIiILw4AOEREREREREZGFYUCHiIiIiIiIiMjCMKBDRERERERERGRhGNAhIiIiIiIiIrIwDOgQEREREREREVkYBnSIiIiIiIiIiCwMAzpERERERERERBaGAR0iIiIiIiIiIgvDgA4RERERERER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ZyqazgXroS7H/ne900pWsJ7lydNKKOwz6jotr5iK7MMVZN1Wkf9+pSyoCtYp7sPbETFnK4a9cRTrRIW3RAV5XpZ1K6BMc0P2zYqW5YqeM+TbfG7U/unMNS2nBsTMHI7x/9uKnrMTxfEu2DLWmhbl2EecseRDxeZT7vwoR54XzruU/Uv+njy/67bfExER0U2DLXSqknyB5le+i1m58nGzB3Nkciuc8lZYDHtW4cEdl9Dkni8x8noEc2QVyAPLxbKY5rpdDMsBlAqkpzMVTueC9SjfDnt900pWsJ7XaRey7OvXcXnFVGQfriDrtpZhQZVeJ2uHvE08gB7tSgnmyAryvDwVbGWaG7JvVrQsV/ScUam8uEauaTm19gMnPx7dtqWPNS2qYIGWfKjYfMqdH+XI88J5l7J/yd9jMIeIiOjfiQEdujZM8fj87fVoYUrEyEeHlPsWIiojpjNVCwZLh7ynk6YhvLkyiqjMlI6NVUCzekVu+CIiIiL6V2NAh64J3XdLMXVTLG57eBvuaa2MpCrHdKbqQYvIyXNwVPch+tr1D0NUqktJ2LM/EbnJwxHo4AlcRERERP9W7EOHqp4xAVs+/wF/G70RPnw8ohx0ZEpVgOlMRP8GF2KxfG0sMhGMvhOGIKyKblclIiIiqu4Y0CEiIiIiIiIiqmZ4yxURERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXDgA4RERERERERUTXD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qlev7ulUtyKDOr5gjly3HCBC1h5hQKfi+AdxAj1KDOZUTgzoEBFVUv6X56KXal66iYiIqDKQtXN8zcNPnz7tGSmpVq1aFdZJck5ODtLS0jzrk5MMLMiADptcVZyiwRoGcqoOBnSIiCq5ki7TvHwTERHRxeYL5vgHdWRNnRo1apxzUEd2fpyamuqpjSObcvmCOb5HOnclBWwYyKkaGNAhIqoieLkmIiKiykben8hJBnR8QR0ZhJEBl7p16551TRo5opVcj1yf7GzZF8TxBXQYcDg/mK9VCwM6REREREREdFZkcdJ/kkEd6ejRo57+bmTzq/IGdWQzrhMnTsBut6N+/fqeef6BHAZ0iLxYT42IiIiIiIjOStEgi5xkYEfWqpG1bE6ePOkJ0JRHZmamp7NluQ65rkDbICIGdIiIiIiIiOgc+AdafM2iDAaDpw8dWctG9oUTbFBHjmglAzoxMTGevnOKNrOSExF5MaBDRERERERE58Q/4OILwMiAjslk8gw5LoM6ZZEjWskpOjraM/Q5gzlEpWNAh4iIiIiIiCpE0eCLrGkjgzOy5o0M1pREBnzkMr6aPVLRdRFRYQzoEBERERER0TnzBV58QRhf7RoZ1JHPZVOqQE2vZF87shaPXEYOd+7/WTlJvkciOoMBHSIiIiIiIqoQ/gEYOfkCM3FxcXA4HEhLSysU1JHPZcfJshNkuYz/Z+Qk+R6JqDCN2+3msOVERERERERUYfyHMpeTHM5cBm1k8CYkJAQ1a9b0DGfua4olX/s6QfYFc/yDOkRUHGvoEBERERERUYUqGpSRgRrZP05UVJRn5Cur1VowolVsbCyDOURnQeNyuVhDh4iIiIiIiCpc0Zo6kgziyICOJDtAjoiI8DxnMIeofDQOh4MBHSIiIiIiIrogigZrfIEeIiofjc1m49lDRERERERERFSFaPbu28+ADhERERERERFRFaJRWL+NiIiIiIiIiKhK4ShXRERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVDAM6RERERERERERVjEYR1OdEREREREREFe79n+z4fLUDKafccLnVmURUQKcF4mtocWdPA8beYFTnlo4BHSIiIiIiIjovDp90444pVqxNcQNOBbXCNdBq1DeJqIBbAdLyxD96DbrHazH/0VA0rFl6oyoGdIiIiIiIiOi8uPI/eVj7twvRUUCoKKgSUemsTgVZ2UD3y3T4/X/h6tzA2IcOERERERERVTjZzErWzGEwhyh48lyR54w8d+Q5VBoGdIiIiIiIiKjCyT5zZDMrBnOIysdzzohzx3MOlYIBHSIiIiIiIqpwsgNk2WcOEZWfPHfkOVQaBnSIiIiIiIiowsnRrNgBMtHZkedOWSPCMaBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDRERERERERFTFMKBDREREREREl7YMN8xHFe+Uoc4rkQvmY+ryx8SkzqWLQeT/cbfnmdkmpjK/u38WjSKoz4mIiIiqNrcV2aezkHk6DzZ9KGIiIxERE4lQnfo+Xfpsp7Bv2y78uWULjqAeOnXogDbtmqCmSX3/Qsg+gLXLV+P3I9niRRTaDvg/3NCk/Dtwctc67PMUXkJQP+FyNLiQaSCqAHXuz1GfXWQymNM3HBtu1CIaCvb/bkXfjxXENVTfL0QGc0Lw5QcmdNQoCFXcGP+ABZ/X0yBOXeK8sop9zdZ4q17UvEDbvNhKTLMM5mgw8fUIvNBavO9WsGahBT2+cCMuVl3kH+DEh5Hqs+IY0CEiIqKqT7Ej4+B2bNx9ClaXC4q4AdeJG0FoNNDpjKhW7zK0at0AMXp1eboE2bBv3vNoP/x9WBEBRJoQiXzk5GSJ9+rj0bk/4bVhTXC+YyK2jVPRo/O/sdFYF7Af8cwbueAkZg6u4XkeLNvGN9G685M4gDDxajC+PPEZbvtHlOzoUlKZAjq4LQJ/36ZDtHztcGPOlFz86zcN4up4lvDjhPl0GFZ/akIPefpZXHj2rjxMqnYBgisysGEyYfq9OtR1uTHlTRtWXupBnVLT7BBTJFZ9aEBzz2vhiB2aB/IRV1+jzrj0lRbQYZMrIiIiquLcyNy7Cd//mQqXKMDrxWuTwQC9wQhF3OrkuxRk7PkZ67aZRZH/DNvxLfjp5yT8tEcW+C8gmxl/yu3+vBeZ6iw6Vzb8+fptaD78G9Som4swmNG4diRycmRhoD7QyIlPhzfFmG+OeRc/X04n4aHOL2Jj/Waoabdh8BNv4M13ZuNf3coXzIFlNZ7v/DoONGqLxs0sogBqUN8gogph0GLE6HA8XEeB2arOK8WFqgGRnw10udeI0d0MGNBRj27/gD9ClJ5mLcxmF1L8fqbTZHCOtW4LMKBDREREVVv2YazblYXq4Q44UA9tevRCn+uvR9/rxWOPy9GmjhFuU1M0alyrUO2MvJxTyMm6CH89zsvBIVHYpwq061MMe3or4urvx5HaT2LWzmwc2LsNivM4UlaMRNtDemQ2AmYN+Qhrz2PWn1y5EB8ba6Bexm7c9MWfWPDGExj32F24tp66QDBcx/DlfYPweo1YROQEUdKs5CZum4RDeSnqq4uD+0ABRejwxjMhwF/uStNHTqYL6FftTM0Tb88xl7ZMJzCs1pmwROE06xAXkY/EiVbM2+3CipX5qD3SDtT559TOKQsDOkRERFSlZaYdRLZWLwrCGjRN6IhGMaHQyzscMS80pi5adroW/W+8Gi1j/G97rLDkiFvFi/DXT6vcMFu8V6g/v/sQu2FE3tG2ePnDl3BbK7V6us6EBr3GYerzotCW2xqx+A5/HvC+dT4c2b9aFBKBHMuduLFHeaI4Z+yb+zT+b14sDKfq4D/P3IGTp89uPRebDF5o5kfjxV+excg/xqpzLyzuA5XF1MSIPdONwK7yXZPNsk+eE4qnk17Pa9lZr6ezZd96xPvHxZQj5slJvqe+42NO83v/uJjEa+zSo3ONM8GK/Dzxj28Z76wSZXg6fRaT/z7J9cr5vnX4OhQ+6TdPpKMwx5n3AnQIXXbazyh9WfGeTPPfejT3tIPzKpRmOSnit/u4A8OfzEPiVDvQpEgTtFyxHtmBdd6Z7fvSVaz2lS/dp8Uk8sKzf+l+nxPPq1qnywzoEBERUdXmdkPe/mrEf3pD4FsbbZH5blsOslJd0Ok1cCluOB1OdSr+91C3LQ+ZJ4/j0L79OHTsFDLznOo7hTlzM5CRnoHsfHWG4kReuvjcoePIsKjrdduRk3VKbFM8Ff84CrbrlH09Bs1tyfFs6+TJU+IxC3m2ovstt+3dn5L218Mp0qYulydbJ/lzWZEt3xPbOCkes31pKMJtzSq8HZGf1mwzUvYdwYkccfN93uUA9e/C82Mvx4Dn38F9CUV7yTEhKlw8iAx2oRaiytmJju3wLqxdsw6/rlwtHv/CvpNqycSfOJ6yT+cgL/+oONjkF6mB3WrzzJOTzeVdrCyy35wb71qDOtiLaz/7HPe3OI5cS9XrCVkGMRqvugKwZAMx8vVh9Z0L55+0D5v+3Iz5X3yNae/N8EzyuZxHJbNZzlxwm18VgtWP6WBOVWeUShT4d7iQOCQM2z6LRP6sSE//Jop4zJwZhuk3ivUcEMscN+C16dFQ5kUhf3YUMj8IFV+UmK+uxXzQheFPis+I95T5UcibFor3xkVg26pQXFVNXShMh1cXeT+fPzscs2UgQn2rKPMBF+r2CUXSTLFswT5F4PBbIbhhUDhSPeuIwG9DxDp2u2EeJ/bdMy8Kaa8Z/YI6MpgTiW3qe8pX4XisIKgjHoNIu495d6BlI5D6kbrsOj3+84zYlkxzdfVDhdLsN30mpq/F9IWaBjXokieDZo1N+GaKNy/lNuSUPzsSeyaFYFh1sewJ77KeYI4v3Z9G4Our3Rg+Ihx7PvX7nJj/zRCt2Pdy/CBfZOwUmYiIiKq0zH0rkbTLjQhkQ1OvG67uWBehJdXGlv3X/LYdx50uWPRq55i2XLj04mYbbuh0zdD1+uay7AV37nHs3rEH+9JtcLuccLlc0Ou00Ip1hza4Ale2L7ydzD1JSNphQWi1dujZMxapf2zA1nQHDI5cKO16o0ven9hsdiPPoUVUiOcTYr3iBlc8c4t1N+/RDy3lhksh92nXtr+xP8MuXji9FX00GujFToXEtULHto0R6yn/u5Hx969Y+bcVRo2CltfcFGDdVqSsW4MNqVaEmpqhW59WiJVxL7WD6Q1/n0S+yw6XU4FGq4VBp0FondZ+2/CyHduCxesPI8zQGJ1vaA5lx1r8nmKD1pWD0IY9cMMVddUlL5YcLHnkSgycdwpIfwC/509E9yBiJLY9C/HCE6/gte9kMxkZ7fKGDYEMXH7XB5j+xv3oXlO8FE5+Mxa1hkwHYtuisWfkle1I3wdxRMpMz8TLGxQ8myDnl8KyGk+HD8brsSeBvguQ8ulghH97L6qP+AXR1mvx4YlPqkynyNetuAm/7lwK1IgCTmXjl9tX49paPdR3L4x/wj5kZWVj2Q8/YePGzahRozpCQjwXFuTn5+PUqXQkJHREv743IDpabP8iqaydImfttmNptBHDfOeUw4WpT+ThkQwN4kKLd4o8/q48vCoDLes0mDhfHXEpIAXJ31jQ82Mn0DwUG94wwhdj3p6Uh3ZT3J714+pIrHtIj3j5hlj/6HsduPPjEO/2AnJj3v15GC6eFb0M5GW4EHt7JNYP1sGv5dIZMhbvmS/27atc9Jwm9mFSJE5c5V04a7cNMeNsiKsr01SkE+KzSftnYoMaGbCKwidiG4EvtwpWzLHDNFjN46CpafhK7KX4znKGRyB1kB61SupmzNf59XrxvdrFfj3pl+7TCkKqaQLsn5qO+ZVnJC0ZbCpJoK+ciIiIqMqIqlYPGre4CTVFQTm6Gr8m/YGt+44jI1DNFIMR4UZRWHfBG8zxUKDT6cTkd0doPY41SZux8XgWtG4nwqJroE5cbUSEGJGnCYFt3y/YsDfLc59ciF4PZ6YBJ/ZuwLojeTBqFbi1esRFRMAUphfb0MLpLXMJCrSe7erEbomdKoM7cz9W/boF29Nt0LnsCI+tiVq1aiI2wgCrSwfHsXVY89tmnPDUENIipmY9sW692Ac7UswZxffVcgpHTrkQorHD0LA2PC3SFCuObl6FH/5Kg9OehYjYhmjVrg1aN60NsafIPrIGa9ft8fTz4E+vN8Cl6JB1YBuW/50Jg0i3ViPSFF6Q2IvDcgxrJ4/BwKl5QLoZIxfcF1wwZ+NUdG15D17bmIWaOInLr++DmwYl4somChDSEpnLHsCVte7DEvXP5VGtu2L0oKGoFeWCxXPYtfUUEho3qy+mtnJG6Xz95sSGi4LncHz52mA00MnBk8s+LiqbQkGMjEoQSLmE90EGc3bu/Bvx8fWRnZMDp9PhmeRzOU++J5ehQBQMfyQfybnqS4MO948PBX53l9y0KU3BkGnheMYX0MhyYdYcC7ren4c7FzmR4qnlqEGPm0LwRkuxzF4LHvzeBV9/vm17huC1xgrM20147zY1mCODBz/k44MD+bj3WwcW/uFCmme+YHNjxR9ynpycWJcaqB6GEzn5ofi8/5lgTtYxB16dmY+xM22Yt0+BrSJK/OVJu0gjdKF4LEEN5jjcWLbIiqFvWTF2kQMb0xXxW+FA4ktWjCo1zWcmcSkuzK4gp3EYtvkFc9L2qen+wo5lx9S8Uju/flw2sfLOKRAt+ypKd2GhyH/5mRUFOyHS0VNcey3FP1MZsYYOERERVXF2pO7YhF/2nEZYiBEmjR0OhwsarQ768JqIb9oCTepHi/nq4lLmXixOPoAQxQK07IUbWvg14PdwI/PAFuw6HY4GLZqifqTa2Y4rC7t/X4dtOfkIR2NccX071Fbf8tTQ2S1Hy85CbqoDdTtdhWa1wz2DcegjYtWmPuLzP2/AnziNWHQoqA1UJiUHB5LXYk22G7EOI5r06IFWNdTCviJu6FO24fu/0lBNSYeucSKua18LevmZVWuxITcPUcbW6Nq7pbcGjir74Br8uD1XpMOAVtf0QhNR7pS1bRZuSEWM9jQiL+uLK0W+FHzEcgQbVu6E2ZWDWlf0R9cGslaT9zNL/0xFuMkOa2YWwptchbaNRX6L97ShMYgJq4jSRDltfBNN7vgUB/edBMLDENOuD/735st48KogRpty7cKULt3xmLkWah2vgdG/LsGL16ifE2n/e9YjaPXAWjSK2Y2Qfy3HuncS4av/8Ofr7dDpNaBmxnY8FkytHNW+z+5E87s2oTr+xoO/5ovteaNOnpo/I36qMjV0ZMe/sq8Y1BQ5kp6NWf0+x91NZJ2CC+efsg+ySdW8eV97AjfrN27G0088istatfS89/eu3XjtzSnoktARKSlHMWzYUHS6oqPnvQut8tbQsSHmXiswMBx7HjWguRoU8My/LR9oEVK8hk6mEUkLQtDb83PhxrJpeej/nlhvG/FyuxtdpkRh1Q06z7Vve1IO2r0vnpwQx8PsCLwgAzyCXP+DaQZ80tMb7PBs724b0E68n+FCXGx4odoxTwzJw1u+vzXUDjB8+Smx/bEROJGoDvuU7oSmSx7QUW5PFPM3afHNukjcWku+eQ41dMqZ9j4pofhrlB6ezaY5oKkt9ul6sT1Z0cTXr029INIsHXBj7vIYtTaVmoYpbjz+QRTeauvNV9sBO0KuEyvurKb7dwO+2RKmphvYuyYPLR5yIm7amXTjtMirrmK/ZN7L5tYtwrHtdQPaet4uuUbUxcAaOkRERHQJM6J2687o16UpaprcsLu0cEIPl9YobhzTsP/PlViZvAOp5RowSIuYJlege6eWZ4I5ki4adeqHQ+MKgeI+iEzfX3f9KDYtohIS0bV1A9SsHotYMZW335ai3BnH8VemgpouC/TN26OlL5gjafSIjG+GrtFaZBmi4DhyFGZZS0cTiTpNqkHjDoeSvxNHT/rVWFJycCrFIt7Lhb5Wc9TxRCTykHrkFLR68eHQjmjdzC+YI4XVQ6OmOuS7jcg8dkosXYTDDlP8NbiyU2PUUdN9UYI5qoP7tot/xfZNRkTt/hAvTfkaa0963yuN7Y9FeOzPmmjk2Ivwp1/Ds75gjqSLxGUjn8K77fNwKLw2jk+eix/P8U+4vn5z6ob/jSte2yK2d44Hy3kw+8BcT38wpZHLFAQx0rJxd+cxFRrE4D4UtmfPPk8zK3NqmieY0759WxgNBs8kn8t58j25jFyWAhDXTPych2f/cMPXM1Z0SxP2TDOgoFqNj0MU7O/Qo50v9p/uxoowA8Y8acKYfmJ6KhSdjinwdcPToIG4RtsVxDVzYeLb9oKaQHL9n3RTa67kuvD8aG8wRwYN4mLdcm4hBtn/lwzkBArmCLKT4WeanBnDO+WQHWgjlo2Vk0hfK7e4Zqtvnq2zSfsfypksrGVA6qEITB6m7qd8kPsnHspMs5jQwBu0KUSrR++Cka4UbNgsMqKztiDdMV0ceG3vmXU3jxffadFN2RSgmVi+upjixPNDbmR6arhWLRfvV5aIiIioomj0iKrXEt2vS8T113ZBQqsGqGVyI8ehhz40Etr0zdiy63jBTfu5MBnDoZF/ASzhNsqtMSC+To0KvcnKzT4l7kU1Yt161KwWVXzdmkjExGmgk4Em5QjyLN7ZoXEN0EzcPDvFp0+YT4lHVZYZW7PdCNdqUbthHXjq2thykJrhhkmrg9NkR9qB/di7z386iNQsGwxGcVecne/5O64/l9uA2g1qIzzAvfcF1/FhZGVkI23/Inz5dA+knG4O089jcWWft/FnGQfBvm2rgCgjHDnAtd1aeQte/nStkDA0BrDUFO/Nxr5D6vyzYVmN5zu/jv11D+H45e/i1UcvL749j4uXqbLp0Mgf7kTj7xqWGMz4NS3Zs4wniHEqG9e27Y9ZXWX1hIrBfSguPT3d02dObLWYgpo5/uQ8+Z5cRi5LgcU10mLBQ3l41a8T3ObdQvDJ1Wdee8hgQIRffyvV9XhrVAjeLzT5mlH5CdUh0mzFuJ/ONL0yeWqfKEheZsG7MpjgmXuWxG7W9OuDJs3sAgwVfL04i7Sbl+djnl+e1mqox6N3hUOZG4ltL4bgBvH7XGwEqvKoJn7zvZVEBQUpsmWhX383IXog5ZhfUClKi2ElV3Lxqgy/XWeBAR0iIiK6dGi0MIXHon6ztuh2XS9c36E2si0uKGFRcJ0w42R5//rmyMPJI3uxddM6JP+ahJ9+TsJ3m2TzojN/Eb0Q3A6ruGUVt20ifeFhfrVz/OiNoTLkA42iQZ5VHV3KUAP16mthUULgPpaKk54ojBsZ5iNwQDZLa4za4ubcQ5EhIwVufQiMp7dj/+492F1kOpwh++QJHHKoVHQmRFWLRM0mXXHbUx9j+7QmSEFr1PlrHD5bWXoTEFv2MZGZWtjFsdKsfuAmWlGx9TxV9EWWY9+xU+rc8srBT//9N16vXgv1HDVx01Um/Pr+FLw1eWrB9O6Psg2fDmE19mPZh+965v10gQdr8ozMJL/yDGD01ieKBTNkEOO6L3uKwlQUkJWNa1v3xy+9v1ffrRjcBzp/NIjr6sTEflbM89W2M2gx6jFj8eCMvyxXsX5e/KcvNoqLbYg3QqCVAZFo8dLzykeD+Hhx7c0qEjg6J4p3yO/zHZgIJu2dFEwckotnZf84/kF0kwZtOxrx41vhePRw1eijprJjQIeIiIguTbLWTnxLXFlfhyx7mLjXTUFesAEdxYnMgxuw9OdkrF6/DUfTMpGHcERFRaJ2hAZ2Oe74ReDZatFq4yq3S941y1s7/33To2a9BuIeWg/FtR/H05xiwQykHpTd7VqhbxqP2n59FchP6u3ZcNW7GlffcD1uvL5PoemGPr1xfWIf9O7VNLi+fyqJNl0TRSHEjhCx0z/tPqDODUKAfrUlW/4pkdXnehtthMkkvjOrC05DNNa+8xDGPf4anvj3pILpfwuOo3GUSez3r/jh9ZfFvEewKYhmYxXphbbPALIGQITIu9+/9jQp8pFBjesW9/R2/JuXjUZ1256XIAb3objq1at7RrPKOJ3p6TOnKDlPvieXkctSafSIu9KG4ZPs2O6rehihRbxfzRfPpTVXOVPL06lg8G0WDJ5kDTiNXqTAM0KSOL+zmoThk2u8/cvIZj5Z6jU8vosJC/oD5kzv67NW0Em9BrXriYcSrltn7azSrkFcdwWTJuWi9p3Z6DPNhnm73QVpR7Qe9z6vAU75NWkujzyxPwXVRDWI7yUe1KHMPeR2RP77WonJ/Z9XSbpzqmgM6BAREdElTOe529EodvFPfYQHWbnEenQzvv9TFNrdeah3+bW47vobcMO13dG9W1d0aVkLDmcJUZXzxBQmSwYukQ43si2B6qm74XC4xb8KFI0OkRFnavFoY+vi8igNLG4FKSdOIz8zHbtcspehamhaN/bMzaApHJEGLWwaBVqLHYpOB71BH3jSV6JbSPNqvPLAdWjVvB2un7FLnVmYLc9bY8ktvrZaIYFrOPnUbNgVyHeKPAd2Hj6mzvVnQ/ZpUbQR5RCb+CraNA2io+WATOh+88MYfkM7dO7aUTy/BTcN6lFourapERl2F5zWJqLAmSDmPYW29dWPXyCy/5cX2r0iR18H6kbhxeRnPbVRZBCj8aorvKUJRzZku71Z3aZ7PlPRuA/FtWjRzDM0eVztWp4OkLdu3Q67w+GZ5HM5T74nl5HLUhlC9Yg8ZME9S840jSrEIC41853Y5nuzug5JL4kvXfaNI/t6kf2+iGunDPjJa4O3HasC804dpj9hRFs1cL53vRUPblZ/P7Ra3PR/YsE9JddUccpLV7p43z9Y4U/8xCWfOhPEj68vfuT2u2HOlZ8R29ljQEwZv3shsolWukif3N9UkaZa3spoBc4q7WIf5EhTDrFusf6kVfkYPiQb168/89tpkssF+CktM82S1YYNBcFtDVrIDD4gPiN/Hq1i27u0eKH5mZq0KccccrFLkrz0EBEREVVZ2YfWYfmvf+JAutrMyJ/1FMwnXDDqrNDIkaZ8dd412lLu7exIP5kFo9EBvakdWjaLld0gFHArAe5Ay0Er1qVxl+/OMjS2JmT/jy6tA6eOnpJ/nCzMcQrHU9zQ6rOhNTZDdf++AtTOkRVtKLSHsrDffAg6e75fZ8gqsVytBjo43VFwZW3DsfTA6XS7zi39Fa6aEVkf7sHfzu1IHjMJXxWNwbiO4dsPXxUFjxBYRXn72lZN1DcCa9AtEb0t+bDFAHM/XYV9RYZox+nV+OKtHFEK2o5TmIgrW6nzz4Ip4X58vvArfFfC9NX4q5B10obsY9dhwmdLxbzXMOAiDLkysd14XNumv6cpEUTh7brfeqL/2uHAPlEKDJNNjIBf+pzfocG5D4XJUasSEjp6AjZyNKv5Xy7AlCnveSb5XM6T78llLtYIV1VNeG0d1k/OwYNrznSSXEi4DTO2+65/GvQeFo6lDxmQKF+2MWDSuAgcnh0FZW44noxWYBbX5C6vhWFkA88HYDM70OJhB+Y+5SjoJNkUZ8CeqXrgcNGLuhCiQfc7jBh/byg+G6aB2dfrsJ84E7Boy5lhv00Nxfo+CsHwNjoMHxKG1RvC0K+a+qaPDLrsdp/5TCM9lj6hRxfoMORf4Vj+pKF4c7Pypn2NATPfj0LmR6EYL0fC0ogfsPZ6DKrh++1TkCqv1XJf/AWRZo8IYMqPTvgaPtZqa8KGN40Y0lC8EHkwcV44RsrnksOFJfOdwAUOhl8oDOgQERFR1WU9jp1bTyNHPq75ASvWbcfuI2ZkpJuRsns7Vv+2FftkTRObC7Wb1UeU717SFAo5QqtVJ8v7J5BqccKafgQncvyDFTq4nBb5h8ACtvT9WLfpOJwGvwhPuYQiVNxcax2RcNrE9tKscFoykHI8J9AfKs8Iq4Nm8Qak2cUN8/HfsGnHcWTnO+F2uz2f37NxC7baNQjNdyH2sgaFhieXQqvHifSKfY7ajn37tDAYFMTUreH9Q2oBLWIbNEd9ReSFSN/etauw41gObA4nnGKyZptxYNOvWPbjcuySQ+VWFqauuO/T64BDlyGu0RzcXv96PDNvNf7ctgt/rlyI/96SgP+bF4/Gmu1I6/AebutRxp+rG/bHk49Vx8mMNmiwYhjueOIb7DCLfLDZkH14HV4dPgIfREag/lFg8Kcj0L2M1VWMAIW9C2xWt/cBWTCUcVOrrL2U7G1ilJGNX26/MEEM7kNh/fregNatL/MMTR4VGQm93uCZ5HM5T74nl6HgxbXSYu6Ywp0kF6gFLBibj1lH1PcMWvQbFIrlH0bixIRQPNNTBzmYEoxa3Hq9Hmjq19TK4cbXc8UB00KLyGqFO0mWnTDPEZcwc6Ye5qN2bPNFWrRiPf8Xglf6GjDixlBMaKrWQPEXq4H5Yyvm7/Ptr0asz4TPJ4Tj8/8zoIc4Vs80TVLJz8y2IckX/Bbb6Tc4DOs+DMfXgw1oGx3gM+VJex89bppqwrCGGkTXM+CVCZHI/ywK+XMi8Ewz74+wzezEbZPFumJFmspIc8AmZFFamBfn4tmC4JsGCb1C8PX/xP78LwwvdFRHEhPXzuQl+Xj4gAZxRYNHlwgGdIiIiKjqCqmGhs2qweDUiZs6E2ynDmLvlk1IXrsZm/ak4LTLgXBbDmJaXItWdf2a2oREo0aMDumOCOisu7Dp1+X4ec0WbN58CNmKETVqRsJu10Kj34/NK5KQ/Ie3U+TFK7bD1Kg+GlhtpQdgSmREtdhwcbOsQGey4cC6Ffjx1/XYvG49DpXavl+P2m06oWctDbJ11ZG5fz1Wr/R20iw/v/WkHdVc2TA0vhrtG8j670WE1UGTBjrk2yIQEpIHjbYJ4uMKh3M8whsgoWtDxORrYFMsOPjnr1iRtAI/iylp1WZsOJQJxWlHdk7eWab//Gg2/DV8MexvHDzUFMZ6f2PWfQPRqX0PdOr9GF7YWA11onfgoPk+LP5+LNqUGYuLxA0vf4JJHQ7jiL0ZDn82DG3r1BX5VhfRjW7F+A0mVDfvwNFRS/HO8NJr+1xKGoXH45frVnubVUiyRkp6Nmb1/fyCBTG4D4VFR0fhjv8bimHDhqJRo3gYDDrPJJ/LefI9uQyVhyj4X+7CxNfsSPZFXARvLECLuK42jOoXoLNfSVwUs8xOvPeRBd3HuTHzGf+mVjaM+BGIixGX2Vgt1r9pwXxfB+cGLYYOF9djWWsnwoHBsxzYbvG+5ZOVrSBc9ulTtMag3N9WTjzWx4oZu4vULLK4sWxxHr4pNsiZ+ExDB4Y/YcWyom29xGcWfpGHVw/4AkRnkfZpTnz/lAUv+i1nMolJRh7Ecim77bj2CQtONBX7IddbVppL+LGJa6zD3GezcdMX4rOn1Zl+stJdmPGxBT2nuxBXW515CdIogvqciIiIqEpy5p3C0YMHsf94FvJdLnG7Km5yxD/GyDpofNllaFI9QL8p1jTs2LIdhzIdnuV1xlDUbNwWbRrHwgQ7Mg5ux6Y9p+CQt0piMkTEoWmb1mJdDhxauw67c9xo1CURLdXegTP3JGGdOoS1//xiFCtSd23F5sNZ4j5VEfupQ2jtpmjfqjFiCw+DUpwi9uvwHuw9ZMYpqy+dGujDaiG+RVM0rR0JvfcPoMW403djxaYjnntjbfwV6H2ZX/85RThzzNi/bw8OpebDpd4q6nQmhNeqh6ZNG6NO5Jk/ddqObcGvO7wjPdVsczWuqFd6HzXnjSsHO76dihcmL8Q3yQfFDJlSJxq3T8C1w5/EuNH90KY8ZVvXKayd9Qbe+nAZvtkg/5Qt8yETl/UYgxFPP4yH+rZCVJHg0J+vt8OQj7zP752/Dc8meJ+frZPfjEXXZ1aLZz3x6ur3cdtFaG5V1MRtkzz9x8iC5d2dx5y3YblLw32oOurcX4l6opV9yqjVF83HFcTJapolkf2wZGkQF6eB+bQ496vJ4IOPAnOqmGfSoEtXAzqJ92BxYbrsH0b2ZxMGT4fI5qNiG/XV7aWJ57X8tyfWcUzMq6e+L5ZFfXUbmS6YHVok9jaieZiCkwecWLBevF9DvO8LJBYj9jdFbDfBgH5NxDpPuzBzjTg4xTbmLo7GMM+KFSR/lYueX3n3z5PGE+L5dUYMqqOBzfcZrdjPfLfIH+1Zpl2mU10uRHy2vcGzfllLKWmNE3tl3zhynf5/UygjzeajYn98eSk+79l/laevoDwxr5M37Sax7b1/OJEk0i6roXr3R5Umlq11Js/jRJ6f4RTp1SFOpkmQsa5KcMn1OPFhyWOuM6BDRERElxS3w+n9g55OL0efLpPbKZZXtNAbAiysuOGUHSBrxPsV3RGw2wmnvHfW6z33z+Xl3W/xJMh0nhVf+lFC/lRWlhxky78M64yIijr3NlG27BzYxHeFsEhUwOqqPE9nwLmHPR0FXyzch6qhUgV0zgNztrgIyyZB4vIYF3MWF/JSmDPFtVf2txZaJPhRGqvibZalF/sTpfEEQubOLCGg4yM7UJZNCNXPBCvotPvWLxcpFBwq7qzS7ONLuyRH2VKfXgoY0CEiIiIiIqIL6lIP6FR2QQV0qNIrLaBThf7UQkRERERERERB0QCyf2ObTU4K3J5qnXQpYQ0dIiIiIiIiqnCsoXPxmY+5UTDEo0PD2jlVEGvoEBEREREREf3DyI6X4yI13onBnEsOAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERERFUMAzpERERERERU4XSitOlW1BdEVC7y3JHnUGkY0CEiIiIiIqIKF19Di7Q8RnSIzoY8d+Q5VBoGdIiIiIiIiKjC3dnTAOg1sDoZ1CEqD885I84dzzlUCgZ0iIiIiIiIqMKNvcGI7vFaZGWrBVQiKpM8V+Q5I88deQ6VhgEdIiIiIiIiOi/mPxqK7pfpkOXUwJyrsE8dohLIc0OeI/JckeeMPHfKolEE9TkRERERERFRhXv/Jzs+X+1Ayik3XG51JhEVkB0gyz5zZDOrsmrm+DCgQ0RERERERERUxbDJFRERERERERFRFcOADhERERERERFRFcOADhERERERERFRlQL8Pw+VQPwgalmfAAAAAElFTkSuQmCC)
# Day 728 July 23 2023 Sunday 💯

import numpy as np
np.random.seed(0)
arr = np.linspace(start = 0, stop = 20, num = 1000)
np.random.choice(arr, 100 )

# Day 729 July 24 2023 Monday 💯

import numpy as np
np.random.sample([2,2])

# Day 730 July 25 2023 Tuesday 💯

import numpy as np
np.random.rand(4, 5)


# Day 731 July 26 2023 Wednesday 💯

import numpy as np
start = 1
stop = 10
size = 1000
np.linspace(start = start, stop = stop, num = size)


# Day 732 July 27 2023 Thursday 💯

from scipy.stats import norm
norm.cdf(5, 6, 4)

# Day 733 July 28 2023 Friday 💯
from scipy.stats import norm
import numpy as np
x_values = np.linspace(4, 7, 10)
norm.pdf(x_values, loc = 0, scale = 1)

# Day 734 July 29 2023 Saturday 💯

import numpy as np
from scipy.stats import norm

x_values = np.linspace(start = 0, stop = 10, num = 1000)
y_values = norm.pdf(x_values, loc = 5, scale = 3)

x_values

y_values

#omg i think i missed the 30th i dont even remember - oh no actually i remember exiting out without saving lol.
# Day 735 July 31 2023 Sunday 💯

import numpy as np
np.linspace(-3, 3, 1000)
# Day 736 August 1 2023 Tuesday 💯

import numpy as np
np.arange(5)

# Day 737 August 2 2023 Wednesday 💯

import numpy as np
np.linspace(20, 50, 100)
# Day 738 August 3 2023 Thursday 💯

import numpy as np
np.random.sample([4, 5])
# Day 739 August 4 2023 Friday 💯

import numpy as np
np.random.rand(4,5)


# Day 740 August 5 2023 Saturday 💯

import numpy as np
np.random.rand(1, 3) 

# Day 741 August 6 2023 Sunday 💯

import numpy as np
choices  = ['Sally', 'Su', 'Suzan']
np.random.choice(choices, size = 4)

# Day 742 August 7 2023 Monday 💯

import numpy as np
np.random.sample([4, 5])
# Day 743 August 8 2023 Tuesday 💯

import numpy as np
np.random.rand(3, 4)

# Day 745 August 10 2023 Thursday 💯

import numpy as np
np.random.rand(5,6)
# Day 746 August 11 2023 Friday 💯

import numpy as np
np.random.sample([1,1])

# Day 747 August 12 2023 Saturday 💯

import numpy as np
np.random.rand(5,4)

# Day 748 August 13 2023 Sunday 💯

import numpy as np
np.linspace(-3, 3, 1000)
# Day 749 August 14 2023 Monday 💯

import numpy as np
np.rand(1,2)

# Day 750 August 15 2023 Tuesday 💯

import numpy as np
np.random.sample([3,2])

# Day 751 August 16 2023 Wednesday 💯

import numpy as np
np.random.rand(1,2)


# Day 752 August 17 2023 Thursday 💯

import numpy as np
np.random.sample([3,3])

# Day 753 August 18 2023 Friday 💯

import numpy as np
np.random.rand(2,2)

# Day 754 August 19 2023 Saturday 💯

import numpy as np
np.linspace(start = -3, stop = 3, num = 1000)

# Day 755 August 20 2023 Sunday 💯

import numpy as np
np.random.sample([3,4])


# Day 756 August 21 2023 Monday 💯

import numpy as np
np.random.rand(1,2)


# Day 757 August 22 2023 Tuesday 💯

import numpy as np
np.random.sample([4,5])


# Day 758 August 23 2023 Wednesday 💯

import numpy as np
np.random.rand(4,5)


# Day 759 August 24 2023 Thursday 💯

import numpy as np
np.random.sample([4,5])



# Day 760 August 25 2023 Friday 💯

import numpy as np
np.random.rand(67, 35)

# Day 761 August 26 2023 Saturday 💯

import numpy as np
np.random.sample([30,30])
# Day 762 August 27 2023 Sunday 💯

import numpy as np
np.random.rand(1,5)

# Day 763 August 28 2023 Monday 💯

import numpy as np
np.random.sample([3,4])


# Day 764 August 29 2023 Tuesday 💯

import numpy as np
np.random.rand(4,5)


# Day 765 August 30 2023 Wednesday 💯

import numpy as np
np.random.sample([4,2])


# Day 766 August 31 2023 Thursday 💯

import numpy as np
np.random.rand(1,5)


# Day 767 September 1 2023 Friday 💯

import numpy as np
np.random.sample([4,3])


# Day 768 September 2 2023 Saturday 💯

import numpy as np
np.random.rand(3,4)
# Day 769 September 3 2023 Sunday 💯

import numpy as np
np.random.sample([1,2])

# Day 770 September 4 2023 Monday 💯

import pandas as pd

df = pd.DataFrame({'ColumnA':['data1', 'data2']})

# Day 771 September 5 2023 Tuesday
import numpy as np
np.randomm.sample([3,4])

# Day 772 September 6 2023 Wednesday
import pandas as pd
 df = pd.DataFrame('ColumnName':{'Data1', 'Data2'})

# Day 773 September 7 2023 Thursday
import numpy as np
np.random.rand(3,5)

# Day 774 September 8 2023 Friday
import numpy as np
np.random.sample([4,4])


# Day 775 September 9 2023 Saturday
import numpy as np
np.random.rand(3,4)

# Day 776 September 10 2023 Sunday
import numpy as np
np.random.sample(4,3)

# Day 777 September 11 2023 Monday
import numpy as np
np.random.rand(3,5)

# Day 778 September 12 2023 Tuesday
import numpy as np
np.random.sample([5,7])

# Day 779 September 13 2023 Wednesday
import numpy as np
np.random.rand(1,5)

# Day 780 September 14 2023 Thursday
import numpy as np
np.random.sample([4,5])

# Day 781 September 15 2023 Friday
import numpy as np
np.random.rand(5,6)


# Day 782 September 16 2023 Saturday
import numpy as np
np.random.sample([4,2])


# Day 783 September 17 2023 Sunday
import numpy as np
np.random.rand(4,6)


# Day 784 September 18 2023 Monday
import numpy as np
np.random.sample([4,3])

# Day 785 September 19 2023 Tuesday

fig, ax = plt.subplots(1,1)
plt.figure(figsize = (20,15))


fig.set_facecolor('none')
ax.set_facecolor('none')
ax.hist(df['Fairly Symmetrical'], color = "#01B99F", edgecolor = 'black')
ax.set_xlabel('Ages')
ax.set_ylabel("Frequency")
ax.spines[['right', 'top']].set_visible(False)

#September 19th 2023

mean = df['Fairly Symmetrical'].mean()
std = df['Fairly Symmetrical'].std()

x_values = np.linspace(mean + 3*std, mean - 3*std,  num = 1000)
y_values = norm.pdf(x_values, mean, std)
ax.plot(x_values, y_values)


fig.set_facecolor('none')
ax.set_facecolor('none')

# Day 785 September 20 2023 Wednesday
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,1)
plt.figure(figsize = (20,15))

df = pd.DataFrame({'Values':[7, 8, 10, 11, 11, 13, 14, 17, 53]})
fig.set_facecolor('none')
ax.set_facecolor('none')
ax.hist(df['Values'], color = "#01B99F", edgecolor = 'black')
ax.set_xlabel('Ages')
ax.set_ylabel("Frequency")
ax.spines[['right', 'top']].set_visible(False)


#Day 786 September 21 2023 Thursday

sample = np.random.uniform(mean, std, 1000)
fig,ax = plt.subplots(1,1)
plt.figure(figsize = (20,15))
fig.set_facecolor('none')
ax.set_facecolor('none')
ax.hist(sample, color = "#01B99F", edgecolor = 'black')
ax.set_xlabel('Ages')
ax.set_ylabel("Frequency")
ax.spines[['right', 'top']].set_visible(False)

# example of a bimodal data sample
from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
# generate a sample
sample1 = normal(loc=20, scale=5, size=300)
sample2 = normal(loc=40, scale=5, size=700)
sample = hstack((sample1, sample2))
# plot the histogram
pyplot.xlabel('Ages')
pyplot.ylabel("Frequency")
pyplot.hist(sample, bins=50, color = "#01B99F")

Making a Dot Plot

https://stackoverflow.com/questions/49703938/how-to-create-a-dot-plot-in-matplotlib-not-a-scatter-plot

import matplotlib.pyplot as plt
import numpy as np
import arviz as az

data = np.array([7,8,10,11,11,13,14,17])

ax = az.plot_dot(data, dotcolor = "#01B99F", dotsize = 0.8)

ax.set_title("Ages of People in The Room")
ax.spines[['right', 'top', 'left']].set_visible(False)

Making a Stem and Leaf Plot

!pip install stemgraphic==0.9.1 -qqq

data = np.array([7,8,10,11,11,13,14,17])


http://stemgraphic.org/doc/modules.html#module-stemgraphic.graphic

import stemgraphic

#create stem-and-leaf plot
fig, ax = stemgraphic.stem_graphic(data, aggregation = False, bar_color = "none", bar_outline = "none", delimiter_color= "#01B99F", median_color = "none", underline_color = "none", trim_blank  = True)


# Day 787 September 22 2023 Friday
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pandas as pd

df = pd.DataFrame({'Outlier Data': [5, 7, 8, 10, 11, 11, 13, 14, 17]})

fig,ax = plt.subplots(1,1)
plt.figure(figsize = (20,15))

fig.set_facecolor('none')
ax.set_facecolor('none')
ax.set_xlabel('Ages')
ax.set_ylabel("Frequency")
ax.spines[['right', 'top']].set_visible(False)
ax.hist(df['Outlier Data'], color = "#01B99F")

ax.set_title("Ages of People in The Room")
ax.spines[['right', 'top', 'left']].set_visible(False)
#Day 788 September 23 2023 Saturday
import numpy as np
np.random.rand(4,5)

# Day 789 September 24 2023 Sunday

import numpy as np 
np.random.sample([5,4])



# Day 790 September 25 2023 Monday

import numpy as np 
np.random.rand(2,4)

# Day 791 September 26 2023 Tuesday
import numpy as np 
np.random.sample([6,7])

# Day 792 September 27 2023 Wednesday
import seaborn as sns
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
df = pd.DataFrame({'Ages':[7, 8, 10, 11, 11, 13, 14, 17]})
fig, ax = plt.subplots(figsize=(5, 3))

sns.set(style="darkgrid")
sns.boxplot(x = df.Ages, color = "#01B99F", flierprops={"marker": "o"}, medianprops={"color": "white"}, ax = ax)

plt.show()






# Day 793 September 28 2023 Thursday
import numpy as np
np.rand(1,5)


# Day 794 September 30 2023 Saturday
import numpy as np
np.sample([5.4])

# Day 795 October 01 2023 Sunday
import numpy as np
np.random.sample([4,5])


# Day 796 October 02 2023 Monday
import numpy as np
np.random.random(42,54)



# Day 797 October 03 2023 Tuesday
import numpy as np
np.random.sample([8,2])

# Day 798 October 04 2023 Wednesday
import numpy as np
np.random.rand(4,5)

# Day 799 October 05 2023 Thursday
import numpy as np
np.random.sample([2,4])


# Day 800 October 06 2023 Friday
import numpy as np
np.random.rand(5,6)


# Day 801 October 07 2023 Saturday
import numpy as np
np.random.sample([5,1])

# Day 802 October 08 2023 Sunday
import numpy as np
np.random.rand(6,7)

# Day 803 October 09 2023 Monday
import numpy as np
np.random.sample([5,3])

# Day 803 October 10 2023 Tuesday
import numpy as np
np.random.rand(3,4)

# Day 804 October 11 2023 Wednesday
import numpy as np
np.random.sample([5,2])

# Day 805 October 12 2023 Thursday
import numpy as np
np.random.sample([5,2])

# Day 806 October 13 2023 Friday
import numpy as np
np.random.rand(6,7)

# Day 807 October 14 2023 Saturday
import numpy as np
np.random.sample([3,1])

# Day 808 October 15 2023 Sunday
import numpy as np
np.random.rand(3,4)


# Day 808 October 16 2023 Monday
import numpy as np
np.random.sample([3,5])

# Day 809 October 17 2023 Tuesday
import numpy as np
np.random.rand(4,3)

# Day 810 October 18 2023 Wednesday
import numpy as np
np.random.sample([5,6])

# Day 811 October 19 2023 Thursday
import numpy as np
np.random.sample([5,3])

# Day 812 October 20 2023 Friday
import numpy as np
np.random.sample([10,40])


# Day 813 October 21 2023 Saturday
import numpy as np
np.random.rand(5,6)


# Day 814 October 22 2023 Sunday
import numpy as np
np.random.rand(5,3)

# Day 815 October 23 2023 Monday
import numpy as np
np.random.sample(3,3)

# Day 816 October 24 2023 Tuesday
import numpy as np
np.random.sample(5,9)

# Day 817 October 25 2023 Wednesday
import numpy as np
np.random.sample(8,5)

# Day 818 October 26 2023 Thursday
import pandas as pd

df = pd.DataFrame({'Column': [6,4,2]})

# Day 819 October 26 2023 Friday
import numpy as np
np.random.sample([4,3])

# Day 820 October 27 2023 Saturday
import numpy as np
np.random.sample([5,7])

# Day 821 October 29 2023 Sunday
import numpy as np
np.random.sample([6,7])

# Day 822 October 30 2023 Monday
import numpy as np
np.random.rand(4,5)

# Day 833 October 31 2023 Tuesday
import numpy as np
np.random.sample(36,6)

# Day 834 November 01 2023 Wednesday
import numpy as np
np.random.rand(3,2)

# Day 835 November 02 2023 Thursday
import numpy as np
np.random.sample([3,2])

# Day 836 November 03 2023 Friday
import numpy as np
np.random.rand(3,2)


# Day 837 November 04 2023 Saturday
import numpy as np
np.random.sample([3,4])

# Day 838 November 05 2023 Sunday
import numpy as np
np.random.rand(5,3)

# Day 839 November 06 2023 Monday
import numpy as np
np.random.sample([4,7])


# Day 840 November 07 2023 Tuesday
import numpy as np
np.random.rand(5,2)


# Day 841 November 08 2023 Wednesday
import numpy as np
np.random.sample([5,3])

# Day 842 November 09 2023 Thursday
import numpy as np
np.random.rand(4,5)

# Day 843 November 10 2023 Friday
import numpy as np
np.random.sample([6,4])

# Day 844 November 11 2023 Saturday
import numpy as np
np.random.rand(45,6)

# Day 845 November 12 2023 Sunday
import numpy as np
np.random.sample(5,3)


# Day 846 November 13 2023 Monday
import numpy as np
np.random.rand(4,3)

# Day 847 November 14th 2023 Tuesday
import numpy as np
np.random.sample([6,4])

# Day 848 November 15th 2023 Wednesday
import numpy as np
np.random.rand(5,2)


# Day 849 November 16th 2023 Thursday
import numpy as np
np.random.sample(4,7)


# Day 850 November 17th 2023 Friday
import numpy as np
np.random.rand(3,7)


# Day 851 November 18th 2023 Saturday
import numpy as np
np.random.sample(4,3)



# Day 852 November 19th 2023 Sunday
import numpy as np
np.random.rand(4,5)

# Day 853 November 20th 2023 Monday
import numpy as np
np.random.sample([3,5])

# Day 854 November 21st 2023 Tuesday
import numpy as np
np.random.rand(7,4)

# Day 855 November 22nd 2023 Wednesday
import numpy as np
np.random.sample(7,4)


# Day 856 November 23rd 2023 Thursday
import numpy as np
np.random.rand(4,9)

# Day 857 November 24th 2023 Friday
import numpy as np
np.random.sample([7,4])

# Day 858 November 25th 2023 Saturday
import numpy as np
np.random.rand(4,3)

# Day 859 November 26th 2023 Sunday
import numpy as np
np.random.sample([5,7])

# Day 860 November 27th 2023 Monday
import numpy as np
np.random.rand(6,8)

# Day 861 November 28th 2023 Tuesday
import numpy as np
np.random.sample([6,6])

# Day 862 November 29th 2023 Wednesday
import numpy as np
np.random.rand(3,6)

# Day 863 November 30th 2023 Thursday
import numpy as np
np.random.sample([5,3])


# Day 864 December 1st 2023 Friday
import numpy as np
np.random.sample([12,35])


# Day 865 December 2nd 2023 Sat
import numpy as np
np.random.sample([5,10])

# Day 866 December 3rd 2023 Sun
import numpy as np
np.random.sample([1,3])


# Day 867 December 4th 2023 Mon
import numpy as np
np.random.rand(11,3)


# Day 868 December 5th 2023 Tue
import numpy as np
np.random.sample(1,2)

# Day 869 December 6th 2023 Wed
import numpy as np
np.random.rand(10,10)

# Day 870 December 7th 2023 Thur
import numpy as np
np.random.rand(2,34)

# Day 871 December 8th 2023 Fri
import numpy as np
np.random.rand(1,1)

# Day 872 December 9th 2023 Sat
import numpy as np
np.random.rand(11,12)

# Day 873 December 10th 2023 Sun
import numpy as np
np.random.rand(12,28)

# Day 874 December 11th 2023 Mon
import numpy as np
np.random.rand(2,3)

# Day 875 December 12th 2023 Tue
import numpy as np
np.random.sample([3,2])

# Day 876 December 13th 2023 Wed
import numpy as np
np.random.sample([5,2])


# Day 877 December 14th 2023 Thur
import numpy as np
np.random.sample([1,2])




# Day 878 December 15th 2023 Fri
import numpy as np
np.random.sample([3,3])

# Day 879 December 16th 2023 Sat
import pandas as pd
nums = [3,5,5]
pd.Series(nums).mean()


# Day 880 December 17th 2023 Sun

import pandas as pd
from itertools import combinations

population = [1,3,2]
from itertools import product

population = [4,7,15]
comb = product(population, repeat=3)  # generate combinations with repetition of length 3
for triplet in list(comb):
    print(triplet)
import pandas as pd
from itertools import product
from fractions import Fraction

population = [4, 7, 15]
comb = product(population, repeat=3)  # generate combinations with repetition of length 3

# Create an empty DataFrame to store the combinations
df = pd.DataFrame(columns=['Combination'])

# Iterate through the combinations and calculate the means
for triplet in comb:
    mean = Fraction(sum(triplet), len(triplet))
    df = pd.concat([df, pd.DataFrame({'Combination': [triplet], 'Mean': [mean]})], ignore_index=True)

# Print the DataFrame
print(df)

import pandas as pd
from fractions import Fraction

# Assuming you already have a DataFrame named 'df' with 'Mean' column
mean_counts = df['Mean'].value_counts().reset_index()
mean_counts.columns = ['Mean', 'Count']

# Assuming you already have a DataFrame named 'mean_counts' with 'Mean' and 'Count' columns
mean_counts['p(x)'] = Fraction(1, 3)**3

# Convert 'Count' column to fractions
mean_counts['Count'] = mean_counts['Count'].apply(lambda x: Fraction(x))

# Calculate 'p(x) * Count' as fractions
mean_counts['p(x) * Count'] = mean_counts['p(x)'] * mean_counts['Count']

# Sort the DataFrame by 'Mean' column in ascending order
mean_counts = mean_counts.sort_values(by='Mean')

# Print the updated DataFrame
print(mean_counts)

import pandas as pd
from fractions import Fraction

# Assuming you already have a DataFrame named 'mean_counts' with 'Mean', 'Count', and 'p(x) * Count' columns
results = mean_counts[['Mean', 'p(x) * Count']].copy()

# Convert 'p(x) * Count' column to fractions
results['p(x) * Count'] = results['p(x) * Count'].apply(lambda x: Fraction(x))

# Print the 'results' DataFrame
results

# Day 881 December 18th 2023 Mon
print("I need finals to be over")

# Day 882 December 19th 2023 Tue
print("I also need sleep")

# Day 883 December 20th 2023 Wed
print("3 down 2 to go")

# Day 884 December 21st 2023 Thur
print("4 down 1 to go")

# Day 885 December 22nd 2023 Fri
print("5 down 0 to go")

# Day 886 December 23rd 2023 Sat
print("At this point, actually no- the past 4 months - actually no this whole notebook is pathetic. I need to get back to it.")

# Day 887 December 24th 2023 Sun
import pandas as pd
df = pd.DataFrame({'COLA':[1,313,333]})

# Day 888 December 25th 2023 Mon
df = pd.DataFrame({'Class':['Calc 135', 'Calc 152', 'Intro to Comp Sci', 'Data Structures']})


# Day 889 December 26th 2023 Tue
import numpy as np
np.add([3,5,2],[1,3,2])

# Day 890 December 27th 2023 Wed
import numpy as np
np.random.sample([3,2])

# Day 891 December 28th 2023 Thur
import numpy as np
np.random.rand(2,4)

# Day 892 December 29th 2023 Fri
import numpy as np
np.random.sample([4,2])

# Day 893 December 30th 2023 Sat
import numpy as np
np.random.rand(2,4)

# Day 894 December 31st 2023 Sun
import numpy as np
np.random.sample([5,2])
print("also happy new year, k bye")

# Day 895 January 1st 2024 Mon
import numpy as np
np.random.rand(1,1)

# Day 896 January 2nd 2024 Tue
import numpy as np
np.random.sample([5,6])


# Day 897 January 3rd 2024 Wed
import numpy as np
np.random.rand(2,3)


# Day 898 January 4th 2024 Thur
import numpy as np
np.random.rand(2,5)

# Day 899 January 5th 2024 Fri
import numpy as np
np.random.rand(1,7)

# Day 900 January 6th 2024 Sat
import numpy as np
np.random.sample([5,5])

# Day 901 January 7th 2024 Sun
import numpy as np
np.random.sample([3,3])

# Day 902 January 8th 2024 Mon
import numpy as np
np.random.rand([2,3])

# Day 903 January 9th 2024 Tue
import numpy as np
np.random.sample([1,5])


# Day 904 January 10th 2024 Wed
import numpy as np
np.random.sample([4,6])

# Day 905 January 11th 2024 Thur
import numpy as np
np.random.sample([4,2])

# Day 906 January 12th 2024 Fri
import numpy as np
np.random.rand(3,4)


# Day 907 January 13th 2024 Sat
import numpy as np
np.random.sample([1,1])

# Day 908 January 14th 2024 Sun
import numpy as np
np.random.sample([3,5])

# Day 909 January 15th 2024 Mon
import numpy as np
np.random.sample([3,4])

# Day 910 January 16th 2024 Tue
import numpy as np
np.random.rand(2,3)

# Day 911 Janurary 17th 2024 Wed
import numpy as np
np.random.sample([2,4])

# Day 912 Janurary 18th 2024 Thur
import numpy as np
np.random.rand(1,4)



# Day 913 Janurary 19th 2024 Fri
import numpy as np
np.random.sample([3,3])


# Day 914 Janurary 20th 2024 Sat
import numpy as np
np.random.rand((5,7))



# Day 915 Janurary 21st 2024 Sun
import numpy as np
np.random.sample(5,3)


# Day 916 Janurary 22nd 2024 Mon
import numpy as np
np.random.rand(3,4)

# Day 917 January 23rd 2024
import pandas as pd
df = pd.DataFrame({'Col1'['data1',  'data2']})

# Day 918  January 24th 2024
import numpy as np
np.random.rand(3,4)

# Day 919  January 25th 2024
import numpy as np
np.random.sample([3,4])

# Day 920  January 26th 2024
import numpy as np
np.random.sample([2,1])

# Day 921  January 27th 2024
import numpy as np
np.random.rand(3,4)

# Day 922  January 29TH 2024
import numpy as np
np.random.sample([3,1])

# Day 923  January 30th 2024
import numpy as np
np.random.rand(4,5)

# Day 924  January 31st 2024
import numpy as np
np.random.rand(4,8)

# Day 925 February 1st 2024
import numpy as np
np.random.sample([3,3])

# Day 926 February 2nd 2024
import numpy as np
np.random.sample([3,2])


# Day 924  February 3rd 2024
import numpy as np
np.random.rand(8,5)


# Day 925  February 4th 2024
import numpy as np
np.random.rand(2,3)


# Day 926  February 5th 2024
import numpy as np
np.random.sample([3,6])


# Day 927  February 6th 2024
import numpy as np
np.random.rand(3,4)

# Day 928  February 7th 2024
import numpy as np
np.random.rand(4,6)

# Day 929  February 8th 2024
import numpy as np
np.random.sample([3,3])

# Day 930  February 9th 2024
import numpy as np
np.random.rand(2,4)

# Day 931 February 10th 2024
import numpy as np
np.random.sample([4,6])

# Day 932 February 11th 2024
import numpy as np
np.random.rand(3,4)

# Day 933 February 12th 2024
import numpy as np
np.random.sample([3,4])

# Day 934 February 13th 2024
import numpy as np
np.random.rand(2,2)

# Day 935 February 14th 2024
print("i bubble u")

# Day 936 February 15th 2024
import numpy as np
np.random.sample([3,3])

# Day 937 February 16th 2024
import numpy as np
np.random.rand(2,5)

# Day 938 February 17th 2024
import numpy as np
np.random.sample([3,5])

# Day 939 February 18th 2024
import numpy as np
np.random.rand(5,3)


# Day 940 February 19th 2024
import numpy as np
np.random.sample([3,2])

# Day 941 February 20th 2024
import numpy as np
np.random.rand(2,3)

# Day 942 February 21st 2024
import numpy as np
np.random.sample([3,2])
