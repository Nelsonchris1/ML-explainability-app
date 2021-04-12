import streamlit as st
from explain import pdplot


st.write("""
    # My First App
    #
""")

train_X = st.file_uploader("X_train")
st.write("""
    #
     """)

test_X = st.file_uploader("X_test")
st.write("""
    #
     """)

y_train = st.file_uploader("y_train")
st.write("""
    #
     """)

y_test = st.file_uploader("y_test")
st.write("""
    #
     """)

model = st.file_uploader("model")
st.write("""
    #
     """)

allFiles = [train_X, test_X, y_train, y_test, model]

if  st.button("Submit uploaed files"):
    cleared = False
    for eachFile in allFiles:
        if not eachFile:
            st.warning("All files need to e uploaded before submitting")
            cleared = True
            break
        st.write("model interpreted")
        
    
