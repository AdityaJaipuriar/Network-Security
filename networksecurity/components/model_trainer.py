import os,sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_array_data,evaluate_models
from networksecurity.utils.main_utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.main_utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

class Modeltrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def train_model(self,x_train,y_train,x_test,y_test):
        models = {
            "Random Forest":RandomForestClassifier(verbose=1),
            "Decision Tree":DecisionTreeClassifier(),
            "AdaBoost":AdaBoostClassifier(),
            "Logistic Regression":LogisticRegression(verbose=1),
            "Naive Bayes":GaussianNB()
        }
        params = {
            "Decision Tree":{
                'criterion':['gini','entropy','log_loss'],
                #'splitter':['best','random'],
                #'max_features':['sqrt','log2']
            },
            "Random Forest":{
                #'criterion':['gini','entropy','log_loss'],
                #'max_features':['sqrt','log2'],
                'n_estimators':[8,16,32,64,93]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,0.01,0.5,.001],
                'n_estimators':[8,16,32,64,93]
            },
            "Naive Bayes":{}
        }
        model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
        
        #best_model_score = max(sorted(model_report.values()))
        #best_model_name = max(sorted(model_report.keys()))[
        #list(model_report.values()).index(best_model_score)
        #]
        # Use the max() function with the 'key' argument to find the best model's name
        best_model_name = max(model_report, key=model_report.get)
        # Use the name to get the best model's score
        best_model_score = model_report[best_model_name]

        # Use the correct string name to retrieve the model object
        best_model = models[best_model_name]
        #best_model = models[best_model_score]

        # CORRECTED: Fit the best model before using it for prediction
        best_model.fit(x_train, y_train)
        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        # Track the MLflow

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)

        #Model Trainer Artifact
        model_trainer_model=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,train_metric_artifact=classification_train_metric,test_metric_artifact=classification_test_metric)

        logging.info(f"Model trainer artifact : {model_trainer_model}")
        return model_trainer_model

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact
            


        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
