# Enhancing DILI Toxicity Prediction by Combining Graph Attention (GATNN) and Dense Neural Network (DNN)

In this file, we would like to share how to use code files in every folder repository. The diagram block of the model is shown in the picture below.
We created an ensemble that connected two models with different methods in parallel processing. Every output of the model was concatenated in a classification layer.
The colored block in the DNN sub-model means that the model is used as a pre-trained model and is trained and saved first before it is joined to the GATNN model. All weights in the DNN model 
would be frozen and only weights in the GATNN and Classification layer would be trainable. 

<img src="https://github.com/user-attachments/assets/59c50365-91ac-426e-8ffe-473c2dd5128a" width="800" />

# Manual 
We coded the model using the Jupyter Notebook platform. All the models were created based on the Pytorch and some data features were generated by using RDkit library packages. 
The environment was set by Anaconda and it can be seen in the **"dili_environment.txt"** file. 
 - python==3.11.4
 - numpy==1.25.0
 - pandas==1.5.3
 - rdkit==2023.3.2
 - scikit_learn==1.3.0
 - torch==2.0.1
 - torch_geometric==2.3.1
 - torchvision==0.15.2

The dataset was given in the folder "Data_preprocessing": **"DILI-Toxicity.csv"**

# Data_preprocessing
In this folder, we made data preprocessing to repair, select, reduce, and find the most important data feature.
the file **"generate_graph_and_fusion_preprocessing.ipynb"** will generate the finally clean data. 
all data would be saved as graph-structured data, reduced fingerprint data, and reduced descriptor data. 

# Ablation Studies 
We have made many experiments before we got the best model. These codes in the "Ablation Studies" folder were utilized to run the DNN model with different input features and the GATNN model in standalone mode. 
 - **"Eksperiment_ECFP2.ipynb"**
 - **"Eksperiment_FUSION.ipynb"**
 - **"Eksperiment_MACCS.ipynb"**
 - **"Eksperiment_MORDRED.ipynb"**
 - **"Eksperiment_PUBCHEM.ipynb"**
 - **"Eksperiment_GRAPHDATA.ipynb"**

these are the results from the ablation studies :
![image](https://github.com/user-attachments/assets/4b5bd516-a3be-473f-b765-25df7237a3a9)
![image](https://github.com/user-attachments/assets/550bb71a-38be-45a0-88ff-9a9fddbce0a7)
![image](https://github.com/user-attachments/assets/7bf34954-f681-4f53-8436-b8a3a48d5a23)
![image](https://github.com/user-attachments/assets/6a980ff4-d21e-44b2-bb9c-9909c184ee5a)
![image](https://github.com/user-attachments/assets/b6302b47-72dd-4379-b5c8-d3b7b4d2b033)
![image](https://github.com/user-attachments/assets/b2d1d2ba-e46c-4483-901e-720c03296807)

# Proposed Model 
As mentioned before, we proposed the ensemble model. This model was realized in these code :  
-**"hybrid_model_graph_MACCS_2.ipynb"** 
-**"performance_result.ipynb"**
the first code is used as optimization for the ensemble model by using the Optuna whereas the second code is used to show its performances of the best model resulted from the optimization. 
this is the performance result of the ensemble model. 

 ![image](https://github.com/user-attachments/assets/fc29a734-6677-4d68-abec-9624fb498096)
![image](https://github.com/user-attachments/assets/f0949018-9e33-44d8-9e01-d939f865b357) ![image](https://github.com/user-attachments/assets/30b441ec-004c-4bf7-b9ee-e89f690e7f4c)

