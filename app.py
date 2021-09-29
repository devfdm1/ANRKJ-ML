
import streamlit as st
from multiapp import MultiApp
import home
from classification import ClassificationMain
#from clustering import main
 # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Predict Loan Risk", ClassificationMain.main)
#app.add_app("Clustering",main)

#app.add_app("Model", model.app)
# The main app
app.run()