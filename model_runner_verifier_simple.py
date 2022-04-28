# Santiago Nunez-Corrales and Eric Jakobsson
# Illinois Informatics and Molecular and Cell Biology
# University of Illinois at Urbana-Champaign
# {nunezco,jake}@illinois.edu

# A simple tunable model for COVID-19 response
from batchrunner_local import BatchRunnerMP
from multiprocessing import freeze_support
from covidmodelcheckpoint_simple import CovidModel
from covidmodelcheckpoint_simple import CovidModel
from covidmodelcheckpoint_simple import Stage
from covidmodelcheckpoint_simple import AgeGroup
from covidmodelcheckpoint_simple import SexGroup
from covidmodelcheckpoint_simple import ValueGroup
from covidmodelcheckpoint_simple import *
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import numpy as np
import concurrent.futures
import multiprocessing
import os
import glob
import copy
import matplotlib.patches as mpatches
from os.path import exists


directory_list = []
filenames_list = []
virus_data_file = open(str(sys.argv[1])) #First arguement must be the location of the variant data file


for argument in sys.argv[2:]:   # Every following arguement must be a folder containing scenario data
    directory_list.append(argument)

for directory in directory_list:    #Searches through the directories for scenario files
    file_list = glob.glob(f"{directory}/*.json")
    for file in file_list:
        filenames_list.append(file)

# Read JSON file
data_list = []
for file_params in filenames_list:      #Creates a data list based on the filenames
    with open(file_params) as f:
        data = json.load(f)
        data_list.append(data)

indexes = [range(len(data_list))]       #Creates a list of indeces associating an index to a data set.
virus_data = json.load(virus_data_file)

def runModelScenario(data, index, iterative_input): #Function that runs a specified scenario given parameters in data.

    print(f"Location: { data['location'] }")
    print(f"Description: { data['description'] }")
    print(f"Prepared by: { data['prepared-by'] }")
    print(f"Date: { data['date'] }")
    print("")
    print("Attempting to configure model from file...")
    # Observed distribution of mortality rate per age
    age_mortality = {
        AgeGroup.C80toXX: data["model"]["mortalities"]["age"]["80+"],
        AgeGroup.C70to79: data["model"]["mortalities"]["age"]["70-79"],
        AgeGroup.C60to69: data["model"]["mortalities"]["age"]["60-69"],
        AgeGroup.C50to59: data["model"]["mortalities"]["age"]["50-59"],
        AgeGroup.C40to49: data["model"]["mortalities"]["age"]["40-49"],
        AgeGroup.C30to39: data["model"]["mortalities"]["age"]["30-39"],
        AgeGroup.C20to29: data["model"]["mortalities"]["age"]["20-29"],
        AgeGroup.C10to19: data["model"]["mortalities"]["age"]["10-19"],
        AgeGroup.C00to09: data["model"]["mortalities"]["age"]["00-09"],
    }

    # Observed distribution of mortality rage per sex
    sex_mortality = {
        SexGroup.MALE: data["model"]["mortalities"]["sex"]["male"],
        SexGroup.FEMALE: data["model"]["mortalities"]["sex"]["female"],
    }

    age_distribution = {
        AgeGroup.C80toXX: data["model"]["distributions"]["age"]["80+"],
        AgeGroup.C70to79: data["model"]["distributions"]["age"]["70-79"],
        AgeGroup.C60to69: data["model"]["distributions"]["age"]["60-69"],
        AgeGroup.C50to59: data["model"]["distributions"]["age"]["50-59"],
        AgeGroup.C40to49: data["model"]["distributions"]["age"]["40-49"],
        AgeGroup.C30to39: data["model"]["distributions"]["age"]["30-39"],
        AgeGroup.C20to29: data["model"]["distributions"]["age"]["20-29"],
        AgeGroup.C10to19: data["model"]["distributions"]["age"]["10-19"],
        AgeGroup.C00to09: data["model"]["distributions"]["age"]["00-09"],
    }

    # Observed distribution of mortality rage per sex
    sex_distribution = {
        SexGroup.MALE: data["model"]["distributions"]["sex"]["male"],
        SexGroup.FEMALE: data["model"]["distributions"]["sex"]["female"],
    }

    # Value distribution per stage per interaction (micro vs macroeconomics)
    value_distibution = {
        ValueGroup.PRIVATE: {
            Stage.SUSCEPTIBLE: data["model"]["value"]["private"]["susceptible"],
            Stage.EXPOSED: data["model"]["value"]["private"]["exposed"],
            Stage.INFECTED: data["model"]["value"]["private"]["asymptomatic"]+data["model"]["value"]["private"]["sympdetected"],
            Stage.RECOVERED: data["model"]["value"]["private"]["recovered"],
            Stage.DECEASED: data["model"]["value"]["private"]["deceased"]
        },
        ValueGroup.PUBLIC: {
            Stage.SUSCEPTIBLE: data["model"]["value"]["public"]["susceptible"],
            Stage.EXPOSED: data["model"]["value"]["public"]["exposed"],
            Stage.INFECTED: data["model"]["value"]["public"]["asymptomatic"] + data["model"]["value"]["public"]["sympdetected"],
            Stage.RECOVERED: data["model"]["value"]["public"]["recovered"],
            Stage.DECEASED: data["model"]["value"]["public"]["deceased"]
        }
    }
    model_params = {
        "num_agents": data["model"]["epidemiology"]["num_agents"],
        "width": data["model"]["epidemiology"]["width"],
        "height": data["model"]["epidemiology"]["height"],
        "repscaling": data["model"]["epidemiology"]["repscaling"],
        "kmob": data["model"]["epidemiology"]["kmob"],
        "age_mortality": age_mortality,
        "sex_mortality": sex_mortality,
        "age_distribution": age_distribution,
        "sex_distribution": sex_distribution,
        "prop_initial_infected": data["model"]["epidemiology"]["prop_initial_infected"],
        "rate_inbound": data["model"]["epidemiology"]["rate_inbound"],
        "avg_incubation_time": data["model"]["epidemiology"]["avg_incubation_time"],
        "avg_recovery_time": data["model"]["epidemiology"]["avg_recovery_time"],
        "proportion_asymptomatic": data["model"]["epidemiology"]["proportion_asymptomatic"],
        "proportion_severe": data["model"]["epidemiology"]["proportion_severe"],
        "prob_contagion": data["model"]["epidemiology"]["prob_contagion"],
        "proportion_beds_pop": data["model"]["epidemiology"]["proportion_beds_pop"],
        "proportion_isolated": data["model"]["policies"]["isolation"]["proportion_isolated"],
        "day_start_isolation": data["model"]["policies"]["isolation"]["day_start_isolation"],
        "days_isolation_lasts": data["model"]["policies"]["isolation"]["days_isolation_lasts"],
        "after_isolation": data["model"]["policies"]["isolation"]["after_isolation"],
        "prob_isolation_effective": data["model"]["policies"]["isolation"]["prob_isolation_effective"],
        "social_distance": data["model"]["policies"]["distancing"]["social_distance"],
        "day_distancing_start": data["model"]["policies"]["distancing"]["day_distancing_start"],
        "days_distancing_lasts": data["model"]["policies"]["distancing"]["days_distancing_lasts"],
        "proportion_detected": data["model"]["policies"]["testing"]["proportion_detected"],
        "day_testing_start": data["model"]["policies"]["testing"]["day_testing_start"],
        "days_testing_lasts": data["model"]["policies"]["testing"]["days_testing_lasts"],
        "day_tracing_start": data["model"]["policies"]["tracing"]["day_tracing_start"],
        "days_tracing_lasts": data["model"]["policies"]["tracing"]["days_tracing_lasts"],
        "new_agent_proportion": data["model"]["policies"]["massingress"]["new_agent_proportion"],
        "new_agent_start": data["model"]["policies"]["massingress"]["new_agent_start"],
        "new_agent_lasts": data["model"]["policies"]["massingress"]["new_agent_lasts"],
        "new_agent_age_mean": data["model"]["policies"]["massingress"]["new_agent_age_mean"],
        "new_agent_prop_infected": data["model"]["policies"]["massingress"]["new_agent_prop_infected"],
        "stage_value_matrix": value_distibution,
        "test_cost": data["model"]["value"]["test_cost"],
        "alpha_private": data["model"]["value"]["alpha_private"],
        "alpha_public": data["model"]["value"]["alpha_public"],
        "day_vaccination_begin": data["model"]["policies"]["vaccine_rollout"]["day_vaccination_begin"],
        "day_vaccination_end": data["model"]["policies"]["vaccine_rollout"]["day_vaccination_end"],
        "effective_period": data["model"]["policies"]["vaccine_rollout"]["effective_period"],
        "effectiveness": data["model"]["policies"]["vaccine_rollout"]["effectiveness"],
        "distribution_rate": data["model"]["policies"]["vaccine_rollout"]["distribution_rate"],
        "cost_per_vaccine":data["model"]["policies"]["vaccine_rollout"]["cost_per_vaccine"],
        "vaccination_percent": data["model"]["policies"]["vaccine_rollout"]["vaccination_percent"],
        "step_count": data["ensemble"]["steps"],
        "load_from_file": data["model"]["initialization"]["load_from_file"],
        "loading_file_path": data["model"]["initialization"]["loading_file_path"],
        "starting_step":data["model"]["initialization"]["starting_step"],
        "agent_storage" : data["output"]["agent_storage"],
        "model_storage": data["output"]["model_storage"],
        "agent_increment": data["output"]["agent_increment"],
        "model_increment": data["output"]["model_increment"],
        "vector_movement" : False
    }

    #Adds variant data into the model in the form of a list.
    virus_param_list = []
    for virus in virus_data["variant"]:
        virus_param_list.append(virus_data["variant"][virus])
    model_params["variant_data"] = virus_param_list
    var_params = {"dummy": range(25,50,25)}

    num_iterations = data["ensemble"]["runs"]
    num_steps = data["ensemble"]["steps"]

    batch_run = BatchRunnerMP(
        CovidModel,
        nr_processes=num_iterations,
        fixed_parameters=model_params,
        variable_parameters=var_params,
        iterations=num_iterations,
        max_steps=num_steps,
        model_reporters={
                    "Step": compute_stepno,
                    "CummulPrivValue": compute_cumul_private_value,
                    "CummulPublValue": compute_cumul_public_value,
                    "CummulTestCost": compute_cumul_testing_cost,
                    "Rt": compute_eff_reprod_number,
                    "Employed": compute_employed,
                    "Unemployed": compute_unemployed
                },
        display_progress=True)

    print("Parametrization complete:")
    print("")
    print("")
    print(f"Executing an ensemble of size {num_iterations} using {num_steps} steps with {num_iterations} machine cores...")

    #Will now return a dictionary containing [iteration:[model_data, agent_data]]
    cm_runs = batch_run.run_all()

    # Extracting data into distinct dataframes
    model_ldfs = []
    agent_ldfs = []
    i = 0
    for cm in cm_runs.values():
        cm[0]["Iteration"] = i
        cm[1]["Iteration"] = i
        model_ldfs.append(cm[0])
        agent_ldfs.append(cm[1])
        i = i + 1

    model_dfs = pd.concat(model_ldfs)
    agent_dfs = pd.concat(agent_ldfs)
    model_save_file = data["output"]["model_save_file"]
    agent_save_file = data["output"]["agent_save_file"]

    # TODO-create the nomenclature for the nature of the save file for both model and agent data. (Very important for organizing test runs for different policy evaluations)
    #Iterative input can be used to directly name the model of interest.
    model_dfs.to_csv(model_save_file)
    agent_dfs.to_csv(agent_save_file)

    print(f"Simulation {index} completed without errors.")

class DiffEq():
    def __init__(self, data):
        self.dimensional_contacts = 3  # Testing average daily contacts
        self.N = data["model"]["epidemiology"]["num_agents"]
        self.beta = self.N * self.dimensional_contacts * data["model"]["epidemiology"]["prob_contagion"]
        self.beta_rand = np.random.uniform(low=self.beta-0.25, high=self.beta+0.25, size=(int(data["ensemble"]["steps"] / 96)+1,))
        self.sigma = 1/data["model"]["epidemiology"]["avg_incubation_time"]
        self.gamma = 1/data["model"]["epidemiology"]["avg_recovery_time"]
        self.mortality = data["model"]["epidemiology"]["proportion_severe"]
        self.recovery = 1 - self.mortality
        self.parameters = []
        self.d_0 = 0
        self.r_0 = 0
        self.e_0 = data["model"]["epidemiology"]["prop_initial_infected"]
        self.i_0 = 0
        self.s_0 = 1 - self.e_0
        self.x_0 = np.array([self.s_0,  self.e_0,self.i_0, self.r_0, self.d_0])
        self.timespan = np.arange(0, int(data["ensemble"]["steps"] / 96)+1  , 1)




    def F_simple_varying_R(self, days,variables):
        s, e, i, r, d = variables
        return [-self.gamma * self.beta_rand[int(days)] * s * i,  # ds/dt = -γR₀si
                self.gamma * self.beta_rand[int(days)] * s * i - self.sigma * e,  # de/dt =  γR₀si -σe
                self.sigma * e - self.gamma * (self.mortality * i + self.recovery * i),  # di/dt =  σe -γi
                self.gamma * self.recovery * i,  # dr/dt =  γ*pr*i
                self.gamma * self.mortality * i]  # dd/dt =  γ*pd*i

    def F_simple(self, days, variables):
        s, e, i, r, d = variables
        return [-self.gamma * self.beta * s * i,  # ds/dt = -γR₀si
                self.gamma * self.beta * s * i - self.sigma * e,  # de/dt =  γR₀si -σe
                self.sigma * e - self.gamma * (self.mortality * i + self.recovery * i),  # di/dt =  σe -γi
                self.gamma * self.recovery * i,  # dr/dt =  γ*pr*i
                self.gamma * self.mortality * i  # dd/dt =  γ*pd*i
            ]

    def solve(self):
        self.solution = solve_ivp(self.F_simple, [0,int(data["ensemble"]["steps"] / 96)+1], self.x_0, t_eval=self.timespan)
        #recreating the tables based on the count of agents in the diffeq model
    def solve_rand(self):
        self.solution_rand = solve_ivp(self.F_simple_varying_R, [0, int(data["ensemble"]["steps"] / 96)+1], self.x_0,
                                       t_eval=self.timespan)

    def plot_constant(self, output):
        plt.figure(figsize=(20, 10))
        plt.title("Differential Equation Constant R")
        plt.xlabel('days')
        plt.ylabel('Prop Population')
        plt.plot(self.timespan, self.solution.y[0], color="blue", label="Susceptible")
        plt.plot(self.timespan, self.solution.y[1], color="purple", label="Exposed")
        plt.plot(self.timespan, self.solution.y[2], color="red", label="Infected")
        plt.plot(self.timespan, self.solution.y[3], color="green", label="Recovered")
        plt.plot(self.timespan, self.solution.y[4], color="black", label="Deceased")
        plt.savefig(output.replace("datatype", "SIRD_constant"),dpi=700)

    def plot_random(self, output):
        plt.figure(figsize=(20, 10))
        plt.title("Differential Equation Random R")
        plt.xlabel('days')
        plt.ylabel('Prop Population')
        plt.plot(self.timespan, self.solution_rand.y[0], color="blue", label="Susceptible")
        plt.plot(self.timespan, self.solution_rand.y[1], color="purple", label="Exposed")
        plt.plot(self.timespan, self.solution_rand.y[2], color="red", label="Infected")
        plt.plot(self.timespan, self.solution_rand.y[3], color="green", label="Recovered")
        plt.plot(self.timespan, self.solution_rand.y[4], color="black", label="Deceased")
        plt.savefig(output.replace("datatype", "SIRD_random"),dpi=700)
        plt.close()

    def plot_diff(self, output):

        plt.figure(figsize=(20, 10))
        plt.title("Constant R - Random R")
        plt.xlabel('days')
        plt.ylabel('Difference')
        plt.plot(self.solution.t, np.abs(self.solution.y[0]- self.solution_rand.y[0]), color="blue", label="Susceptible")
        plt.plot(self.solution.t, np.abs(self.solution.y[1]- self.solution_rand.y[1]), color="purple", label="Exposed")
        plt.plot(self.solution.t, np.abs(self.solution.y[2]- self.solution_rand.y[2]), color="red", label="Infected")
        plt.plot(self.solution.t, np.abs(self.solution.y[3]- self.solution_rand.y[3]), color="green", label="Recovered")
        plt.plot(self.solution.t, np.abs(self.solution.y[4]- self.solution_rand.y[4]), color="black", label="Deceased")
        plt.savefig(output.replace("datatype", "SIRD_diff"),dpi=700)
        plt.close()

    def plot_diff_abm(self, abm_data, output):
        self.solution_rand = solve_ivp(self.F_simple_varying_R, [0, int(data["ensemble"]["steps"] / 96)], self.x_0, t_eval=self.timespan)
        plt.figure(figsize=(20, 10))
        plt.title("Differential - ABM Data")
        plt.xlabel('days')
        plt.ylabel('Difference')
        plt.plot(self.solution_rand.y[0] - abm_data["Susceptible"], color="blue", label="Susceptible")
        plt.plot(self.solution_rand.y[1] - abm_data["Exposed"], color="purple", label="Exposed")
        plt.plot(self.solution_rand.y[2] - abm_data["Infected"], color="red", label="Infected")
        plt.plot(self.solution_rand.y[3] - abm_data["Recovered"], color="green", label="Recovered")
        plt.plot(self.solution_rand.y[4] - abm_data["Deceased"], color="black", label="Deceased")
        plt.savefig(output.replace("datatype", "diff_abm"),dpi=700)
        plt.close()

    def plot_diff_abm_const(self, abm_data, output):
        self.solution_rand = solve_ivp(self.F_simple, [0, int(data["ensemble"]["steps"] / 96)], self.x_0, t_eval=self.timespan)
        plt.figure(figsize=(20, 10))
        plt.title("Differential - ABM Data")
        plt.xlabel('days')
        plt.ylabel('Difference')
        plt.plot(self.solution.y[0] - abm_data["Susceptible"], color="blue", label="Susceptible")
        plt.plot(self.solution.y[1] - abm_data["Exposed"], color="purple", label="Exposed")
        plt.plot(self.solution.y[2] - abm_data["Infected"], color="red", label="Infected")
        plt.plot(self.solution.y[3] - abm_data["Recovered"], color="green", label="Recovered")
        plt.plot(self.solution.y[4] - abm_data["Deceased"], color="black", label="Deceased")
        plt.savefig(output.replace("datatype", "diff_abm_const"),dpi=700)
        plt.close()

    def plot_abm(self, abm_data, output):
        plt.figure(figsize=(20, 10))
        plt.title("")
        plt.xlabel('days')
        plt.ylabel('Prop Population')
        plt.title("ABM data")
        plt.plot( abm_data["Susceptible"], color="blue",label="Susceptible")
        plt.plot(abm_data["Exposed"], color="purple", label="Exposed")
        plt.plot(abm_data["Infected"], color="red", label="Infected")
        plt.plot(abm_data["Recovered"], color="green", label="Recovered")
        plt.plot(abm_data["Deceased"], color="black", label="Deceased")
        plt.savefig(output.replace("datatype", "ABM_data"))
        plt.close()
    def plot_R(self, abm_data, output):
        plt.figure(figsize=(20, 10))
        plt.xlabel('days')
        plt.ylabel('R_0')
        plt.title("R(t)")
        print(len(abm_data["R_0"]))
        plt.plot(   abm_data["R_0"], color="blue",label="Susceptible")
        plt.savefig(output.replace("datatype", "R"),dpi=700)
        plt.close()


    def calculate_error(self, model_data, hyperparam_weights):
        error = 0
        for step, value in enumerate(self.solution_rand.y[0]):
            error+= hyperparam_weights[0] * (self.solution_rand.y[0][step] - model_data["Susceptible"][step])**2
            error+= hyperparam_weights[1] * (self.solution_rand.y[1][step] - model_data["Exposed"][step])**2
            error+= hyperparam_weights[2] * (self.solution_rand.y[2][step] - model_data["Infected"][step])**2
            error+= hyperparam_weights[3] * (self.solution_rand.y[3][step] - model_data["Recovered"][step])**2
            error+= hyperparam_weights[4] * (self.solution_rand.y[4][step] - model_data["Deceased"][step])**2
        return np.sqrt(error)

    #Runs through an iterated parameter sweep for a constant relative to R_0 and returns the value
    def optimize(self, model_data, step_size, success_threshold, hyperparam_weights):
        scale = 1
        min_error = 9999999
        step = step_size
        increase = True  # Begin with this value at true since R_0 is guaranteed to be an underestimate
        change = False
        escaped = False
        previous_error = 999999
        iteration = 0
        new_error = 100
        while min_error > success_threshold:
            R_t = []
            # Shift the parameter up and down
            if increase:
                if change:
                    step = step / 2

                scale += step
                for index, item in enumerate(model_data["R_0"]):
                    R_t.append(item * (scale))
            else:
                if change:
                    step = step / 2

                scale -= step
                for index, item in enumerate(model_data["R_0"]):
                    R_t.append(item * (scale))
            # run the differential equation model and calculate the error
            self.beta_rand = R_t
            self.solve_rand()

            previous_error = new_error
            new_error = self.calculate_error(model_data, hyperparam_weights)
            diff = np.abs(new_error - previous_error)

            # This approximation is the best so far
            # If this approximation is better then we continue traversing at the same speed

            if new_error < min_error:
                min_error = new_error
                min_scale = scale
                min_increase = increase
                continue

            # This is not a better estimate, so change directions and change velocity
            if change != True:
                change = True

            if (new_error > previous_error):
                increase = not (increase)  # Change directions

            if (diff) < 0.001:  # This approximation is going nowhere
                scale = min_scale
                increase = not (min_increase)
                change = False
                step = step_size

            iteration += 1
            # print("New_error: ", new_error, "Min_error: ", min_error, "success_threshold: ", success_threshold, "Escaped: ", escaped, "Iterations: ", iteration, "Scaler: ", scale)
            if (iteration > 100):  # Stuck in the loop, error diverged
                escaped = True
                break
        return min_scale, min_error

    def calculate_error_const(self, model_data, hyperparam_weights):
        error = 0
        for step, value in enumerate(self.solution_rand.y[0]):
            error+= hyperparam_weights[0] * (self.solution.y[0][step] - model_data["Susceptible"][step])**2
            error+= hyperparam_weights[1] * (self.solution.y[1][step] - model_data["Exposed"][step])**2
            error+= hyperparam_weights[2] * (self.solution.y[2][step] - model_data["Infected"][step])**2
            error+= hyperparam_weights[3] * (self.solution.y[3][step] - model_data["Recovered"][step])**2
            error+= hyperparam_weights[4] * (self.solution.y[4][step] - model_data["Deceased"][step])**2
        return np.sqrt(error)

    def optimize_const(self, model_data, step_size, success_threshold, hyperparam_weights):
        scale = 1
        min_error = 9999999
        step = step_size
        increase = True  # Begin with this value at true since R_0 is guaranteed to be an underestimate
        change = False
        escaped = False
        R_const_static = average(model_data["R_0"])
        iteration = 0
        previous_error = 100
        new_error = 100
        min_scale = 0
        while min_error > success_threshold:
            # Shift the parameter up and down
            R_const = R_const_static
            if increase:
                if change:
                    step = step / 1.25

                scale += step
                R_const = R_const * scale
            else:
                if change:
                    step = step / 1.25

                scale -= step
                R_const = R_const * scale
            # run the differential equation model and calculate the error
            self.beta = R_const
            self.solve()
            previous_error = new_error
            new_error = self.calculate_error_const(model_data, hyperparam_weights)
            diff = np.abs(new_error - previous_error)

            # This approximation is the best so far
            # If this approximation is better then we continue traversing at the same speed

            if new_error < min_error:
                min_error = new_error
                min_scale = scale
                min_increase = increase
                continue

            # This is not a better estimate, so change directions and change velocity
            if change != True:
                change = True

            if (new_error > previous_error):
                increase = not (increase)  # Change directions

            if (diff) < 0.001:  # This approximation is going nowhere
                scale = min_scale
                increase = not (min_increase)
                change = False
                step = step_size

            iteration += 1
            # print("New_error: ", new_error, "Min_error: ", min_error, "success_threshold: ", success_threshold, "Escaped: ", escaped, "Iterations: ", iteration, "Scaler: ", scale)
            if (iteration > 100):  # Stuck in the loop, error diverged
                escaped = True
                break
        return min_scale, min_error, R_const_static


def average(values):
    mean = sum(values) / len(values)
    variance = sum([((x - mean) ** 2) for x in values]) / len(values)
    res = variance ** 0.5
    count = 0
    count_n = 0
    for item in values:
        if item - res > 0:
            count += item
            count_n += 1
    if count_n == 0:
        return 0
    return count / count_n


#input a plot, a portion of relevant data
#optimize the data as you did before finding the relevant scale
#using the parameters for the data we will create different plots representing the scaled values
def combine_data(data, rand_d, const_d, space, pop, prob, rand_err_d, const_err_d, ravg_d, r_d):
    input_location = data["output"]["model_save_file"]
    output_location = input_location.replace(".csv", "-datatype.png")
    df0 = pd.read_csv(input_location)
    features = ["Susceptible", "Exposed", "Infected", "Recovered", "Deceased", "R_0"]
    full_model_data = {}
    for feature in features:
        df = pd.DataFrame()
        df["Step"] = df0["Step"] / 96
        df[feature] = df0[feature]  # *100
        avg = []
        low_ci_95 = []
        high_ci_95 = []
        for step in df["Step"].unique():
            values = df[feature][df["Step"] == step]
            f_mean = values.mean()
            avg.append(f_mean)

        df_stats = pd.DataFrame()
        df_stats["Step"] = df["Step"].unique()
        df_stats["mean"] = avg
        full_model_data[feature] = df_stats["mean"]
        full_model_data["Step"] = df_stats["Step"]
    agent_count = data["model"]["epidemiology"]["num_agents"]
    model_data = {}
    iteration = 0
    model_data["Step"] = full_model_data["Step"]
    for feature in features:
        model_data[feature] = []
        iteration = 0
        for value in full_model_data[feature]:
            if iteration % 96 == 0:
                if feature == "R_0":
                    model_data[feature].append(value)
                else:
                    model_data[feature].append(value / agent_count)
            iteration += 1
    model_data["R_0"].append(0)
    model_data["R_0"].append(0)
    hyperparams = [0.25, 0.25, 0.25, 0.25, 0]  # Weights for SEIRD in error calculation
    error_threshold = 0.001 * len(model_data["R_0"])  # We want on average to be 0.1 error on every step
    step_size = 0.1  # Step size to increase the R_0 scaling for optimal c*R_0
    diffeqmodel = DiffEq(data)
    scale_rand, err_rand = diffeqmodel.optimize(model_data, step_size, error_threshold, hyperparams)
    scale_const, err_const, r_avg = diffeqmodel.optimize_const(model_data, step_size, error_threshold, hyperparams)
    #We want to return scale_rand and scale_const
    rand_d[space][pop][prob] = scale_rand
    const_d[space][pop][prob] = scale_const
    rand_err_d[space][pop][prob] = err_rand
    const_err_d[space][pop][prob] = err_const
    ravg_d[space][pop][prob] = average(model_data["R_0"])
    r_d[space][pop][prob] = average(model_data["R_0"]) * scale_const

#Here is where we put the model verification process.
if __name__ == '__main__':
    run_models = True
    run_extra = True
    run_tests = False
    space_params = [50,75, 100,125]
    population_params = [500,600,700,800]
    contagtion_params = [0, 0.005, 0.01, 0.025 ,0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 1]
    data_list_dict = {}
    if (run_extra):
        for space in space_params:
            data_list_dict[space] = {}
            for pop in population_params:
                data_list_dict[space][pop] = {}
                for cont in contagtion_params:
                    new_list = copy.deepcopy(data_list[0])
                    new_list["model"]["epidemiology"]["num_agents"] = pop
                    new_list["model"]["epidemiology"]["width"] = space
                    new_list["model"]["epidemiology"]["height"] = space
                    new_list["model"]["epidemiology"]["prob_contagion"] = cont
                    name = new_list["output"]["model_save_file"]
                    new_list["output"]["model_save_file"] = name.replace(".csv", "_"+str(pop)+"_"+str(space)+"_"+str(cont)+".csv")
                    data_list.append(new_list)
                    data_list_dict[space][pop][cont] = new_list

    max_process_count = 96
    total_count = 0
    process_count = 0
    trial_count = len(data_list)-1
    if (run_models == True):
        while (total_count < trial_count):
            processes = []
            print("Count:", total_count, trial_count)
            total_count = 0
            
            process_count = 0
            for space in space_params:
                for pop in population_params:
                    for cont in contagtion_params:
                        data = data_list_dict[space][pop][cont]
                        if process_count < max_process_count:
                            total_count += 1
                            if not (exists(data["output"]["model_save_file"])): #if the trial does not exist
                                print("---------------missing:", space, pop, cont, data["output"]["model_save_file"])
                                process_count += data["ensemble"]["runs"]
                                p = multiprocessing.Process(target=runModelScenario, args=[data, 0, 0])
                                p.start()
                                processes.append(p)
            for process in processes:
                process.join()

    processes = []
    manager = multiprocessing.Manager()
    outputs_rand = manager.dict()
    outputs_const = manager.dict()
    outputs_rand_error = manager.dict()
    outputs_const_error = manager.dict()
    outputs_R_avg = manager.dict()
    outputs_R = manager.dict()
    data_list_iterator = 0
    for space in space_params:
        outputs_rand[space] = manager.dict()
        outputs_const[space] = manager.dict()
        outputs_rand_error[space] = manager.dict()
        outputs_const_error[space] = manager.dict()
        outputs_R_avg[space] = manager.dict()
        outputs_R[space] = manager.dict()
        for pop in population_params:
            outputs_rand[space][pop] = manager.dict()
            outputs_const[space][pop] = manager.dict()
            outputs_rand_error[space][pop] = manager.dict()
            outputs_const_error[space][pop] = manager.dict()
            outputs_R_avg[space][pop] = manager.dict()
            outputs_R[space][pop] = manager.dict()

    max_process_count = 96
    total_count = 0
    process_count = 0
    trial_count = len(data_list)-1
    history = 0
    previous_df_rand = pd.read_csv("scenarios/Verifier/results_rand.csv")
    previous_df_const = pd.read_csv("scenarios/Verifier/results_const.csv")
    while(total_count < trial_count):
        print(total_count, trial_count)
        processes = []
        total_count = 0
        process_count = -1*history
        for space in space_params:
            for pop in population_params:
                for cont in contagtion_params:
                    data = data_list_dict[space][pop][cont]
                    if process_count < max_process_count:
                        total_count += 1
                        #if not(space in previous_df_rand["Space"].unique() and pop in previous_df_rand["Population"].unique() and cont in previous_df_rand["Contagtion"].unique()):
                        if (process_count >= 0):
                            process_count += 1
                            print("Running data for:", space, pop, cont)
                            p = multiprocessing.Process(target=combine_data, args=[data,outputs_rand,outputs_const,space, pop, cont, outputs_rand_error, outputs_const_error, outputs_R_avg, outputs_R])
                            p.start()
                            processes.append(p)
                        else:
                            process_count += 1
                            #outputs_rand[space][pop][cont] = previous_df_rand["Space" == space]["Population" == pop]["Contagtion" == cont]["Scalar"]
                            #outputs_const[space][pop][cont] = previous_df_const["Space" == space]["Population" == pop]["Contagtion" == cont]["Scalar"]
                            #outputs_rand_error[space][pop][cont] = previous_df_rand["Space" == space]["Population" == pop]["Contagtion" == cont]["Error"]
                            #outputs_const_error[space][pop][cont] = previous_df_const["Space" == space]["Population" == pop]["Contagtion" == cont]["Error"]
                            #outputs_R_avg[space][pop][cont] = previous_df_const["Space" == space]["Population" == pop]["Contagtion" == cont]["ABM_R_avg"]
                            #outputs_R[space][pop][cont] = previous_df_const["Space" == space]["Population" == pop]["Contagtion" == cont]["R"]
        history += process_count
        for process in processes:
            process.join()
    
    print("Saving Final Output::::::::::::::::::::::\n\n")
    #Create a dataframe with columns: "Space", "Population", "Contagtion", "Scalar", "Error"
    data_rand = {"Space":[], "Population": [], "Contagtion": [], "Scalar": [], "Error" : []}
    data_const = {"Space": [], "Population": [], "Contagtion": [], "Scalar": [], "Error": [], "ABM_R_avg": [], "R": []}
    for space in space_params:
        for pop in population_params:
            for cont in contagtion_params:
                data_rand["Space"].append(space)
                data_rand["Population"].append(pop)
                data_rand["Contagtion"].append(cont)
                data_rand["Scalar"].append(outputs_rand[space][pop][cont])
                data_rand["Error"].append(outputs_rand_error[space][pop][cont])

                data_const["Space"].append(space)
                data_const["Population"].append(pop)
                data_const["Contagtion"].append(cont)
                data_const["Scalar"].append(outputs_const[space][pop][cont])
                data_const["Error"].append(outputs_const_error[space][pop][cont])
                data_const["ABM_R_avg"].append(outputs_R_avg[space][pop][cont])
                data_const["R"].append(outputs_R[space][pop][cont])

    rand_pd = pd.DataFrame(data=data_rand)
    const_pd = pd.DataFrame(data=data_const)
    rand_pd.to_csv("scenarios/Verifier/results_rand.csv")
    const_pd.to_csv("scenarios/Verifier/results_const.csv")
    #By this stage we should have the parameters good to go
    location = "scenarios/Verifier/"
    colors = ["red", "blue", "green", "brown"]
    cont_list_rand = []

    color_iterator = 0
    for space in space_params:
        #create a new plot
        plt.figure(figsize=(200.7, 100.27))
        plt.ticklabel_format(style='plain', axis='y')
        fig, ax = plt.subplots()
        legends_list = []
        ax.set_xlabel("Prob_contagtion")
        ax.set_ylabel("Scalar_Multiplier")
        for pop in population_params:
            #initialize new subplot, cont_list
            cont_list_rand = []
            for cont in contagtion_params:
                cont_list_rand.append(outputs_rand[space][pop][cont])
            #Plot the subplot
            ax.plot(contagtion_params, cont_list_rand, color=colors[color_iterator], label=("Population"+str(pop)), linewidth=1)
            legend = mpatches.Patch(color=colors[color_iterator])
            color_iterator = (color_iterator+1)%4
            legends_list.append(legend)
            cont_list_rand = []

        #Save the plot here
        ax.set_title("RAND space=(" + str(space)+","+ str(space)+")")
        plt.axis('tight')
        plt.legend(legends_list, population_params, bbox_to_anchor=(0.90, 1.1), loc="upper left", borderaxespad=0, fontsize='xx-small')
        output = "scenarios/Verifier/" + "RAND_space(" + str(space) + ").png"
        plt.savefig(output, dpi=700)
        plt.close()

    cont_list_const = []
    color_iterator = 0
    #Do the same for const
    for space in space_params:
        #create a new plot
        plt.figure(figsize=(200.7, 100.27))
        plt.ticklabel_format(style='plain', axis='y')
        fig, ax = plt.subplots()
        legends_list = []
        ax.set_xlabel("Prob_contagtion")
        ax.set_ylabel("Scalar_Multiplier")
        for pop in population_params:
            #initialize new subplot, cont_list
            cont_list_rand = []
            for cont in contagtion_params:
                cont_list_const.append(outputs_const[space][pop][cont])
            #Plot the subplot
            ax.plot(contagtion_params, cont_list_const, color=colors[color_iterator], label=pop, linewidth=1)
            legend = mpatches.Patch(color=colors[color_iterator])
            color_iterator = (color_iterator+1)%4
            legends_list.append(legend)
            cont_list_const = []

        #Save the plot here
        ax.set_title("CONST space=" + str(space))
        plt.axis('tight')
        plt.legend(legends_list, population_params, bbox_to_anchor=(0.90, 1.1), loc="upper left", borderaxespad=0, fontsize='xx-small')
        output = "scenarios/Verifier/" + "CONST_space(" + str(space) + ").png"
        plt.savefig(output, dpi=700)
        plt.close()





    #Save figures for the error now
    err_list_rand = []
    color_iterator = 0
    for space in space_params:
        # create a new plot
        plt.figure(figsize=(200.7, 100.27))
        plt.ticklabel_format(style='plain', axis='y')
        fig, ax = plt.subplots()
        legends_list = []
        ax.set_xlabel("Prob_contagtion")
        ax.set_ylabel("Error")
        for pop in population_params:
            # initialize new subplot, cont_list
            err_list_rand = []
            for cont in contagtion_params:
                err_list_rand.append(outputs_rand_error[space][pop][cont])
            # Plot the subplot
            ax.plot(contagtion_params, err_list_rand, color=colors[color_iterator], label=("Population" + str(pop)),
                    linewidth=1)
            legend = mpatches.Patch(color=colors[color_iterator])
            color_iterator = (color_iterator + 1) % 4
            legends_list.append(legend)
            err_list_rand = []

        # Save the plot here
        ax.set_title("RAND Error space=(" + str(space) + "," + str(space) + ")")
        plt.axis('tight')
        plt.legend(legends_list, population_params, bbox_to_anchor=(0.90, 1.1), loc="upper left", borderaxespad=0,
                   fontsize='xx-small')
        output = "scenarios/Verifier/" + "RAND_Error_space(" + str(space) + ").png"
        plt.savefig(output, dpi=700)
        plt.close()

    err_list_const = []
    color_iterator = 0
    # Do the same for const
    for space in space_params:
        # create a new plot
        plt.figure(figsize=(200.7, 100.27))
        plt.ticklabel_format(style='plain', axis='y')
        fig, ax = plt.subplots()
        legends_list = []
        ax.set_xlabel("Prob_contagtion")
        ax.set_ylabel("Error")
        for pop in population_params:
            # initialize new subplot, cont_list
            err_list_const = []
            for cont in contagtion_params:
                err_list_const.append(outputs_const_error[space][pop][cont])
            # Plot the subplot
            ax.plot(contagtion_params, err_list_const, color=colors[color_iterator], label=pop, linewidth=1)
            legend = mpatches.Patch(color=colors[color_iterator])
            color_iterator = (color_iterator + 1) % 4
            legends_list.append(legend)
            err_list_const = []

        # Save the plot here
        ax.set_title("CONST Error space=" + str(space))
        plt.axis('tight')
        plt.legend(legends_list, population_params, bbox_to_anchor=(0.90, 1.1), loc="upper left", borderaxespad=0,
                   fontsize='xx-small')
        output = "scenarios/Verifier/" + "CONST_Error_space(" + str(space) + ").png"
        plt.savefig(output, dpi=700)
        plt.close()

    R_list = []
    color_iterator = 0
    # Do the same for const
    for space in space_params:
        # create a new plot
        plt.figure(figsize=(200.7, 100.27))
        plt.ticklabel_format(style='plain', axis='y')
        fig, ax = plt.subplots()
        legends_list = []
        ax.set_xlabel("Prob_contagtion")
        ax.set_ylabel("R")
        for pop in population_params:
            # initialize new subplot, cont_list
            R_list = []
            for cont in contagtion_params:
                R_list.append(outputs_R[space][pop][cont])
            # Plot the subplot
            ax.plot(contagtion_params, R_list, color=colors[color_iterator], label=pop, linewidth=1)
            legend = mpatches.Patch(color=colors[color_iterator])
            color_iterator = (color_iterator + 1) % 4
            legends_list.append(legend)
            R_list = []

        # Save the plot here
        ax.set_title("diffeq R const for space=" + str(space))
        plt.axis('tight')
        plt.legend(legends_list, population_params, bbox_to_anchor=(0.90, 1.1), loc="upper left", borderaxespad=0,
                   fontsize='xx-small')
        output = "scenarios/Verifier/" + "CONST_R_space(" + str(space) + ").png"
        plt.savefig(output, dpi=700)
        plt.close()
    #Done


    # for index, data in enumerate(data_list):
    #     #runModelScenario(data, index, 0)
    #     output_location = data["output"]["model_save_file"]
    #     output_location = output_location.replace(".csv", "-datatype.png")
    #     df0 = pd.read_csv(data["output"]["model_save_file"])
    #     features = ["Susceptible", "Exposed", "Infected", "Recovered", "Deceased", "R_0"]
    #     full_model_data = {}
    #     for feature in features:
    #         df = pd.DataFrame()
    #         df["Step"] = df0["Step"]/96
    #         df[feature] = df0[feature]  # *100
    #         avg = []
    #         low_ci_95 = []
    #         high_ci_95 = []
    #         for step in df["Step"].unique():
    #             values = df[feature][df["Step"] == step]
    #             f_mean = values.mean()
    #             avg.append(f_mean)
    #
    #         df_stats = pd.DataFrame()
    #         df_stats["Step"] = df["Step"].unique()
    #         df_stats["mean"] = avg
    #         full_model_data[feature] = df_stats["mean"]
    #         full_model_data["Step"] = df_stats["Step"]
    #     agent_count = data["model"]["epidemiology"]["num_agents"]
    #     model_data = {}
    #     iteration = 0
    #     model_data["Step"] = full_model_data["Step"]
    #     for feature in features:
    #         model_data[feature] = []
    #         iteration = 0
    #         for value in full_model_data[feature]:
    #             if iteration % 96 == 0:
    #                 if feature == "R_0":
    #                     model_data[feature].append(value)
    #                 else:
    #                     model_data[feature].append(value/agent_count)
    #             iteration += 1
    #     model_data["R_0"].append(0)
    #     model_data["R_0"].append(0)
    #     hyperparams = [0.25,0.25,0.25,0.25,0] #Weights for SEIRD in error calculation
    #     error_threshold = 0.001 * len(model_data["R_0"]) #We want on average to be 0.1 error on every step
    #     step_size = 0.1 #Step size to increase the R_0 scaling for optimal c*R_0
    #     diffeqmodel = DiffEq(data)
    #     scale_rand = diffeqmodel.optimize(model_data, step_size, error_threshold, hyperparams)
    #     scale_const = diffeqmodel.optimize_const(model_data, step_size, error_threshold, hyperparams)
    #     for index, item in enumerate(model_data["R_0"]):
    #         model_data["R_0"][index] = item * (scale_rand)
    #     diffeqmodel.plot_R(model_data, output_location)
    #     diffeqmodel.beta = average(model_data["R_0"]) * scale_const
    #     # Verification process:
    #     #TODO: find a constant parameter that minimizes the error between the agent based model and the differential equation model
    #     print("R_0 rand scalar: ", scale_rand)
    #     print("R_0 const scalar: ", scale_const)
    #
    #     diffeqmodel.beta_rand = model_data["R_0"]
    #     diffeqmodel.solve()
    #     diffeqmodel.solve_rand()
    #     diffeqmodel.plot_abm(model_data, output_location)
    #     diffeqmodel.plot_constant(output_location)
    #     diffeqmodel.plot_random(output_location)
    #     diffeqmodel.plot_diff(output_location)
    #     diffeqmodel.plot_diff_abm(model_data, output_location)
    #     diffeqmodel.plot_diff_abm_const(model_data, output_location)
        # 1. Initialize Differential model for a fixed parameter
        # First visualizing a basic SEIRD model with the following parameters:
            # beta -> constant transmission rate of infected individuals
                # beta will have to be modified to match the spacial dimensions of the model.
                # May have to be the case where B -> B(t)
            # sigma -> average time/ probability of Exposed agents to become infectious
            # gamma -> average time/ probability of Infected agents to become dead/recovered
            # morality -> partition of infected agents that do not recover
            # recovery -> partition of infected agents that recover recovery = (1-mortality)

        # The model will report the following:
            # S -> Susceptible agents
            # E -> Exposed agents
            # I -> Infectious agents
            # R -> Recovered agents
            # D -> Deceased agents



        # 2. Run a parameter sweep for R0 that minimizes the lsq between the two models.
        # 3. Run trial for varying parameters for R_diff.
        # 4. Save the values of R_diff and R_abm and find some relating factor in between them.


        #2. Parameter sweep algorithm
        # For maximumm efficiency we will run trials of #2000 steps in a fixed environment.
        # We will run all 32 iterations in parallel and average the results and their variation values.
        # We evaluate the lsq for the model at that point.
        # Based on the size of the R(w) and whether it was an overestimate or under, we will reshift the prop_contagtion parameter.
        # One thing to note are all the parameters in the OG model and how they affect the overall effect on the model.


