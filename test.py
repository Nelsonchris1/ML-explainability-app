import pandas as pd
import numpy as np
import streamlit as st
np.random.seed(24)
df = pd.DataFrame({'A': np.linspace(1, 10, 10)})

df = pd.concat([df, pd.DataFrame(np.random.randn(10, 4), columns=list('BCDE'))],
               axis=1)
df.iloc[0, 2] = np.nan

def highlight_greaterthan(s,column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column][0:5]
    return ['background-color: yellow' if is_max.any() else '' for v in is_max]


st.dataframe(df.style.apply(highlight_greaterthan, column=['C', 'B'], axis=1))
