import pandas as pd
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def load_data():
    data = pd.read_csv('SchoolsDropoutCleaned (1).csv')
    
    # Remove rows starting from index 4426
    data = data.iloc[:4426]
    
    return data

def bin_data(data, columns_to_bin):
    labels = ['low', 'average', 'high']
    for column in columns_to_bin:
        # Check if the column contains grades (non-zero values)
        if (data[column] != 0).any():
            lower_boundary = data[data[column] != 0][column].quantile(0.25)
            upper_boundary = data[data[column] != 0][column].quantile(0.75)
            print(lower_boundary, upper_boundary)
            bins = [-np.inf, lower_boundary, upper_boundary, np.inf]
            data[column] = pd.cut(data[column], bins=bins, labels=labels)
        else:
            # If all values are zero, assign a default label (e.g., 'low')
            data[column] = 'low'  
    return data


#give column
def calculate_probabilities(data, column):
    probabilities = data[column].value_counts(normalize=True).sort_index()
    #print(probabilities)
    cpd_values = [[p] for p in probabilities.tolist()]
    return cpd_values



#same function of creation of cpds for all cpds
def create_cpd(variable, variable_card, values):
    cpd = TabularCPD(variable=variable, variable_card=variable_card, values=values)
    return cpd

def calculate_conditional_probabilities(data, column1, column2):
    cpd_values = []
    for value in ['low', 'average', 'high']:
        filtered_data = data[data[column1] == value]
        probabilities = filtered_data[column2].value_counts(normalize=True)
        for value in ['low', 'average', 'high']:
            if value not in probabilities.index:
                probabilities[value] = 0
        probabilities = probabilities.sort_index()
        cpd_values.append(probabilities.tolist())
        cpd_values_transposed = list(map(list, zip(*cpd_values)))
    return cpd_values_transposed

def calculate_target_probabilities(data, column1, column2, target_column):
    target_labels = ['Dropout', 'Enrolled', 'Graduate']
    cpd_values = []

    for value1 in ['low', 'average', 'high']:
        for value2 in ['low', 'average', 'high']:
            filtered_data = data[(data[column1] == value1) & (data[column2] == value2)]
            target_probabilities = filtered_data[target_column].value_counts(normalize=True)

            # Reindex to include all possible target labels
            target_probabilities = target_probabilities.reindex(target_labels, fill_value=0)

            cpd_values.append(target_probabilities.tolist())

    # Transpose the resulting list
    cpd_values_transposed = list(map(list, zip(*cpd_values)))

    return cpd_values_transposed

def create_target_cpd(variable, variable_card, values, evidence, evidence_card):
    cpd = TabularCPD(variable=variable, variable_card=variable_card, 
                     values=values, evidence=evidence, evidence_card=evidence_card)
    return cpd

def create_bayesian_model(cpd_approved1, cpd_grade1, cpd_approved2, cpd_grade2, cpd_target):
    model = BayesianNetwork([('Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)'),
                       ('Curricular units 1st sem (grade)3', 'Curricular units 2nd sem (grade)2'),
                       ('Curricular units 2nd sem (approved)', 'Target'),
                       ('Curricular units 2nd sem (grade)2', 'Target')])
    model.add_cpds(cpd_approved1, cpd_grade1, cpd_approved2, cpd_grade2, cpd_target)

    return model

def query_target_probability(infer, evidence_values):
    # Query the conditional probability of 'Target' given the provided evidence
    prob_target_given_evidence = infer.query(variables=['Target'], evidence=evidence_values)

    # Print the result
    print(prob_target_given_evidence)

def Query(model):
    # Create an inference object
    infer = VariableElimination(model)

    # Example: Provide evidence for a specific instance
    evidence_values = {'Curricular units 1st sem (approved)': 'average', 'Curricular units 1st sem (grade)3': 'average'}
    
    # Query the probability of 'Target' given the provided evidence
    query_target_probability(infer, evidence_values)
    

def main():
    data = load_data()
    columns_to_bin = ['Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)3', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)2']
    target_to_bin = ['Target']
    data = bin_data(data, columns_to_bin)
    print(data['Target'].head())
    
    # Curricular units 1st sem (approved) CPD
    cpd_approved1_values = calculate_probabilities(data, 'Curricular units 1st sem (approved)')
    #print(cpd_approved1_values)
    cpd_approved1 = TabularCPD(variable='Curricular units 1st sem (approved)', variable_card=3, values=cpd_approved1_values, state_names={'Curricular units 1st sem (approved)': ['low', 'average', 'high']})

    # Curricular units 1st sem (grade)3 CPD
    cpd_grade1_values = calculate_probabilities(data, 'Curricular units 1st sem (grade)3')
    #print(cpd_grade1_values)
    cpd_grade1 = TabularCPD(variable='Curricular units 1st sem (grade)3', variable_card=3, values=cpd_grade1_values,  state_names={'Curricular units 1st sem (grade)3': ['low', 'average', 'high']})

    # Curricular units 2nd sem (approved) CPD
    cpd_approved2_values = calculate_conditional_probabilities(data, 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)')
    print(cpd_approved2_values)
    cpd_approved2 = TabularCPD(variable='Curricular units 2nd sem (approved)', variable_card=3, values=cpd_approved2_values, 
                           evidence=['Curricular units 1st sem (approved)'],
                           evidence_card=[3], state_names={'Curricular units 1st sem (approved)': ['low', 'average', 'high'] , 'Curricular units 2nd sem (approved)': ['low', 'average', 'high']})


    # Curricular units 2nd sem (grade)2 CPD
    cpd_grade2_values = calculate_conditional_probabilities(data, 'Curricular units 1st sem (grade)3', 'Curricular units 2nd sem (grade)2')
    cpd_grade2 = TabularCPD(variable='Curricular units 2nd sem (grade)2', variable_card=3, values=cpd_grade2_values, evidence=['Curricular units 1st sem (grade)3'],
        evidence_card=[3], state_names={'Curricular units 1st sem (grade)3': ['low', 'average', 'high'], 'Curricular units 2nd sem (grade)2':['low', 'average', 'high']})

    # Target CPD
    cpd_target_values = calculate_target_probabilities(data, 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)2', 'Target')
    cpd_target = TabularCPD(variable='Target', variable_card=3, values=cpd_target_values, 
                                   evidence=['Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)2'],
                                   evidence_card=[3, 3], state_names={'Curricular units 2nd sem (approved)': ['low', 'average', 'high'], 'Curricular units 2nd sem (grade)2':['low', 'average', 'high'], 'Target': ['Dropout', 'Enrolled', 'Graduate']})
    
    model = create_bayesian_model(cpd_approved1, cpd_grade1, cpd_approved2, cpd_grade2, cpd_target)

    Query(model)
    

    

    

    

if __name__ == "__main__":
    main()
