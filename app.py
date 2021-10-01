
import streamlit as st
from multiapp import MultiApp
import home
from classification import ClassificationMain
from clustering import ClusteringMain

# import your app modules here

app = MultiApp()
st.set_page_config(layout="wide")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Predict Loan Risk", ClassificationMain.main)
app.add_app("Bank Marketing Analysis", ClusteringMain.main)

#app.add_app("Model", model.app)
# The main app
app.run()
