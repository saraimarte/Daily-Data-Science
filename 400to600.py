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
