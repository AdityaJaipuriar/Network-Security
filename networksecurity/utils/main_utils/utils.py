import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
def read_yaml_file(file_path:str)-> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        NetworkSecurityException(e,sys)

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array_data(file_path : str,array:np.array):
    """
    Save numpy array to file
    file_path:str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_object(file_path:str,obj:object)->None:
    try:
        logging.info("Entered the save_object method of main_utils")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Exited the save_object method of main_utils")
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exists")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_numpy_array_data(file_path:str)->np.array:
    """
    load numpy array data from file
    file_path: str loaction of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        # 💡 Correctly iterate through the dictionary items (key-value pairs)
        for name, model in models.items():
            
            # 💡 Access parameters using the model's string name as the key
            para = param[name]
            
            # 💡 Create and fit GridSearchCV for the current model
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(x_train, y_train)
            
            # 💡 Use the best_estimator_ from GridSearchCV for predictions
            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(x_train)
            train_model_score = r2_score(y_train, y_train_pred)

            y_test_pred = best_model.predict(x_test)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the score in the report dictionary
            report[name] = test_model_score
        
        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)
