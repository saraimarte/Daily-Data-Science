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



















