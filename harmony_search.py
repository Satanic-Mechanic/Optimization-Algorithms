import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def objective_function(params, X, y, model_choice):
    if model_choice == 'decisiontree':
        model = DecisionTreeClassifier(max_depth=int(params[0]), 
                                       min_samples_split=int(params[1]), 
                                       min_samples_leaf=int(params[2]))
    elif model_choice == 'randomforest':
        model = RandomForestClassifier(n_estimators=int(params[0]),
                                       max_depth=int(params[1]),
                                       min_samples_split=int(params[2]),
                                       min_samples_leaf=int(params[3]))
    elif model_choice == 'svm':
        model = SVC(C=params[0], kernel=params[1])
    elif model_choice == 'neuralnetwork':
        model = MLPClassifier(hidden_layer_sizes=tuple(map(int, params[0])),
                              activation=params[1],
                              learning_rate_init=params[2])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    return val_accuracy

class HarmonySearch:
    def __init__(self, num_variables, variable_ranges, objective_function, max_iterations, prompt_user=False):
        self.num_variables = num_variables
        self.variable_ranges = variable_ranges
        self.objective_function = objective_function
        self.max_iterations = max_iterations
        self.prompt_user = prompt_user

    def optimize(self):
        harmony_memory = self.initialize_harmony_memory(num_harmonies=10)
        best_harmony = harmony_memory[0]
        best_fitness = self.objective_function(best_harmony)
        best_harmonies = []
        best_fitnesses = []

        iteration = 0
        while iteration < self.max_iterations:
            new_harmony = self.generate_new_harmony(harmony_memory)
            new_fitness = self.objective_function(new_harmony)

            if new_fitness > best_fitness:
                best_harmony = new_harmony.copy()
                best_fitness = new_fitness
                best_harmonies.append(best_harmony)
                best_fitnesses.append(best_fitness)

            worst_index = np.argmin([self.objective_function(harmony) for harmony in harmony_memory])
            if new_fitness > self.objective_function(harmony_memory[worst_index]):
                harmony_memory[worst_index] = new_harmony

            iteration += 1

        return best_harmonies, best_fitnesses


    def get_model_choice_and_ranges(self):
        while True:
            model_choice = input("Which model would you like to optimize? Choices: ['DecisionTree', 'RandomForest', 'SVM', 'NeuralNetwork']\n")
            model_choice = model_choice.strip().lower()

            if model_choice == 'decisiontree':
                param_ranges = [(1, 50), (2, 10), (1, 10)]
                break
            elif model_choice == 'randomforest':
                param_ranges = [(10, 200), (1, 50), (2, 10), (1, 10)]
                break
            elif model_choice == 'svm':
                param_ranges = [(0.1, 10), ('linear', 'poly', 'rbf', 'sigmoid')]
                break
            elif model_choice == 'neuralnetwork':
                param_ranges = [[(10, 200), (10, 200), (10, 200)], ('relu', 'tanh', 'logistic'), (0.001, 0.1)]
                break
            else:
                print("Invalid choice, please try again.\n")
        return model_choice, param_ranges

    def user_prompt(self):
        data_path = input("Please input the path to your CSV data file:\n")
        df = pd.read_csv(data_path.strip())
        
        feature_cols = input("Please input the names of the feature columns, separated by commas:\n").split(',')
        target_col = input("Please input the name of the target column:\n")
        
        X = df[feature_cols]
        y = df[target_col]
        
        while True:
            model_choice = input("Which model would you like to optimize? Choices: ['DecisionTree', 'RandomForest', 'SVM', 'NeuralNetwork']\n")
            model_choice = model_choice.strip().lower()

            if model_choice == 'decisiontree':
                param_ranges = [(1, 50), (2, 10), (1, 10)]
                break
            elif model_choice == 'randomforest':
                param_ranges = [(10, 200), (1, 50), (2, 10), (1, 10)]
                break
            elif model_choice == 'svm':
                param_ranges = [(0.1, 10), ('linear', 'poly', 'rbf', 'sigmoid')]
                break
            elif model_choice == 'neuralnetwork':
                param_ranges = [[(10, 200), (10, 200), (10, 200)], ('relu', 'tanh', 'logistic'), (0.001, 0.1)]
                break
            else:
                print("Invalid choice, please try again.\n")

        harmony_search = HarmonySearch(num_variables=len(param_ranges), 
                                       variable_ranges=param_ranges, 
                                       objective_function=lambda params: objective_function(params, X, y, model_choice), 
                                       max_iterations=100)

        best_parameters, best_fitness = harmony_search.optimize()

        print("Best parameters:", best_parameters)
        print("Best fitness:", best_fitness)

    def initialize_harmony_memory(self, num_harmonies):
        harmony_memory = []
        for _ in range(num_harmonies):
            harmony = [random.uniform(range_min, range_max) if isinstance(range_min, (int, float)) else random.choice(range_min) for range_min, range_max in self.variable_ranges]
            harmony_memory.append(harmony)
        return harmony_memory

    def generate_new_harmony(self, harmony_memory):
        new_harmony = []
        for variable_index in range(self.num_variables):
            if random.random() < 0.5:
                new_value = random.uniform(self.variable_ranges[variable_index][0], self.variable_ranges[variable_index][1]) if isinstance(self.variable_ranges[variable_index][0], (int, float)) else random.choice(self.variable_ranges[variable_index])
            else:
                random_harmony = random.choice(harmony_memory)
                new_value = random_harmony[variable_index]
            new_harmony.append(new_value)
        return new_harmony

HarmonySearch(0, [], None, 0, True).user_prompt()

def optimize():
    data_path = input("Please input the path to your CSV data file:\n")
    df = pd.read_csv(data_path.strip())
    
    feature_cols = input("Please input the names of the feature columns, separated by commas:\n").split(',')
    target_col = input("Please input the name of the target column:\n")
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    
    harmony_search = HarmonySearch(num_variables=0, 
                                   variable_ranges=[], 
                                   objective_function=None, 
                                   max_iterations=100)
    model_choice, param_ranges = harmony_search.get_model_choice_and_ranges()

    harmony_search = HarmonySearch(num_variables=len(param_ranges), 
                                   variable_ranges=param_ranges, 
                                   objective_function=lambda params: objective_function(params, X_train, y_train, model_choice), 
                                   max_iterations=100)

    best_parameters, best_fitness = harmony_search.optimize()

    print("Best parameters:", best_parameters)
    print("Best fitness:", best_fitness)

    # Final evaluation of the best model on the test set
def model_based_on_choice(model_choice, best_parameters):
    if model_choice == 'decisiontree':
        model = DecisionTreeClassifier(max_depth=int(best_parameters[0]), 
                                       min_samples_split=int(best_parameters[1]), 
                                       min_samples_leaf=int(best_parameters[2]))
    elif model_choice == 'randomforest':
        model = RandomForestClassifier(n_estimators=int(best_parameters[0]),
                                       max_depth=int(best_parameters[1]),
                                       min_samples_split=int(best_parameters[2]),
                                       min_samples_leaf=int(best_parameters[3]))
    elif model_choice == 'svm':
        model = SVC(C=best_parameters[0], kernel=best_parameters[1])
    elif model_choice == 'neuralnetwork':
        model = MLPClassifier(hidden_layer_sizes=tuple(map(int, best_parameters[0])),
                              activation=best_parameters[1],
                              learning_rate_init=best_parameters[2])

    return model
    best_model.fit(X_train, y_train)
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("Test accuracy of the best model:", test_accuracy)

optimize()