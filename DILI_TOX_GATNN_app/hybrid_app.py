# -*- coding: utf-8 -*-

from flask import Flask, render_template, url_for ,request , redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from hybrid_function import *
import os 
# create empty model with the hyperparameter 



app = Flask(__name__)

picFolder =os.path.join('static','pics')
app.config['UPLOAD_FOLDER']= picFolder
#app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///test.db'

#db = SQLAlchemy(app)


  
    
@app.route('/', methods= ['POST','GET'])

def index():
    pred_result =""
    smiles_input = ""
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'],"blokdiagram2.png")
    if request.method =='POST':
        smiles_input = request.form['content']
        
        pred_result = smiles_to_DILI(smiles_input)
        return render_template('index.html',smiles_input=smiles_input, pred_result=pred_result, user_image=pic1)
        #request.form['result']=pred_result    
    else:
        return render_template('index.html',smiles_input=smiles_input,pred_result=pred_result, user_image=pic1)


    
if __name__=="__main__":
    app.run(debug=True)