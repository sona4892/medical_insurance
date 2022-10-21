import numpy as np
import pickle
import json
#import config

class MedicalInsurance():
    def __init__(self,age, sex, bmi, children, smoker, region):
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region='region_'+region

    def load_model(self):
        with open('Linear_Model.pkl','rb') as f:
            self.model=pickle.load(f)

        
        with open("project_data.json",'r') as f:
            self.json_data=json.load(f)

    def get_predicted_charges(self):
        self.load_model()
        region_index=self.json_data['columns'].index(self.region)

        test_array=np.zeros(len(self.json_data['columns']))
        test_array[0]=self.age
        test_array[1]=self.json_data['sex'][self.sex]
        test_array[2]=self.bmi
        test_array[3]=self.children
        test_array[4]=self.json_data['smoker'][self.smoker]
        test_array[region_index]=1

        print("Test Array: ",test_array)
        predicted_charges=self.model.predict([test_array])
        print(predicted_charges)
        return predicted_charges

if __name__=="__main__":
    age=56
    sex='male'
    bmi=27.9
    children= 4
    smoker='no'
    region='northeast'
    med_ins=MedicalInsurance(age, sex, bmi, children, smoker,region)
    med_ins.get_predicted_charges()