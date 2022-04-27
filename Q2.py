import streamlit as st
import numpy as np
import pandas as pd
import constraint
import math

st.title('Vaccine Distribution Modelling')

states = ['ST-1', 'ST-2', 'ST-3', 'ST-4', 'ST-5']
# max capacity of each state
states_capacity = [5000, 10000, 7500, 8500, 9500]
# population of each state
populations = [[115900, 434890, 15000], [100450, 378860, 35234], [223400, 643320, 22318], 
[269300, 859900, 23893], [221100, 450500, 19284]]
# number of vaccination centres of each state
states_cr_num = [[20, 15, 10, 21, 5], [30, 16, 15, 10, 2], [22, 15, 11, 12, 3], 
[16, 16, 16, 15, 1], [19, 10, 20, 15, 1]]
# dictionary of "max capacity of vaccination centre: name of vaccination centre"
ct_name_dict = {200: 'CR-1', 500: 'CR-2', 1000: 'CR-3', 2500: 'CR-4', 4000: 'CR-5'}

# repeat for all 5 states
for i in range(5):

    # constraint the number of vaccination centres for a state
    cr_prob = constraint.Problem()
    cr_prob.addVariable('CR-1', range(states_cr_num[i][0]+1))  
    cr_prob.addVariable('CR-2', range(states_cr_num[i][1]+1))  
    cr_prob.addVariable('CR-3', range(states_cr_num[i][2]+1))  
    cr_prob.addVariable('CR-4', range(states_cr_num[i][3]+1))
    cr_prob.addVariable('CR-5', range(states_cr_num[i][4]+1))

    # max capacity of the state
    capacity = states_capacity[i]
    # the total max capacity of chosen vaccination centres must be less than or equal to max capacity of state
    def cr_constraint(a, b, c, d, e):
        if a*200 + b*500 + c*1000 + d*2500 + e*4000 <= capacity:
            return True

    cr_prob.addConstraint(cr_constraint, ['CR-1', 'CR-2', 'CR-3', 'CR-4', 'CR-5'])
    solutions = cr_prob.getSolutions() 
      
    max_val = 0
    # find the highest value of total max capacity for the combinations of vaccination centres
    for s in solutions:  
        current_max = s['CR-1']*200 + s['CR-2']*500 + s['CR-3']*1000 + s['CR-4']*2500 + s['CR-5']*4000
        if current_max > max_val:
            max_val = current_max

    best_sol = {}
    minCost = 0
    # find the combination of vaccination centres where the rental cost per day is the lowest & total max capacity is the highest
    for s in solutions:
        if (s['CR-1']*200 + s['CR-2']*500 + s['CR-3']*1000 + s['CR-4']*2500 + s['CR-5']*4000) == max_val:
            currentCost = s['CR-1']*100 + s['CR-2']*250 + s['CR-3']*500 + s['CR-4']*800 + s['CR-5']*1200
            if (minCost == 0) or (currentCost < minCost):
                minCost = currentCost
                best_sol = s

    st.subheader(states[i])
    st.write('Vaccination centres: CR-1: ', best_sol['CR-1'], ' CR-2: ', best_sol['CR-2'], ' CR-3: ', best_sol['CR-3'], ' CR-4: ', 
    best_sol['CR-4'], ' CR-5: ', best_sol['CR-5'])
    st.write("Rental cost per day (RM): ", minCost)

    # dictionary of "max capacity of vaccination centre: number" 
    ct_num_dict = {200: best_sol['CR-1'], 500: best_sol['CR-2'], 1000: best_sol['CR-3'], 2500: best_sol['CR-4'], 4000: best_sol['CR-5']}
    ct_capa_list = []
    # append the max capacity of each vaccination centre type if at least 1 of that centre is used
    for k, v in ct_num_dict.items():
        if v > 0:
            ct_capa_list.append(k)

    total_vacA = 0
    total_vacB = 0
    total_vacC = 0
    # repeat for each vaccination centre type
    while len(ct_capa_list) > 0: 
        # total capacity of the centre is reduced to 4% to speed up the CSP process
        total_capa = int(ct_capa_list[0] * 0.04)
        # probability of vaccine A is set to be more than 0.05 of total capacity
        vacA_Prob = total_capa * 0.05
        # probability of vaccine C is set to be more than 0.25 of total capacity
        vacC_Prob = total_capa * 0.25

        # vaccine A > 0.05 of total capacity, vaccine C > 0.25 of total capacity, vaccine B > vaccine C 
        # total amount of all vaccine types should equal to total capacity of the centre 
        def vac_constraint(a, b, c):
            if a > vacA_Prob and c > vacC_Prob and b > c and (a + b + c) == total_capa:
                return True

        # constraint the number of each vaccine types for a vaccination centre
        vac_prob = constraint.Problem()
        vac_prob.addVariable('>60', range(total_capa))  
        vac_prob.addVariable('35-60', range(total_capa))  
        vac_prob.addVariable('<35', range(total_capa)) 

        vac_prob.addConstraint(vac_constraint, ['>60', '35-60', '<35'])
        solutions = vac_prob.getSolutions() 

        vacc_sol = {}
        max_vacc = 0
        # find the number of each vaccine types distributed to a centre per day by taking the highest value for age 35-60 (majority)
        for s in solutions:  
            current_max = s['35-60']
            if current_max > max_vacc:
                max_vacc = current_max
                vacc_sol = s
      
        # resume all vaccine numbers by multiplying by 25.
        # previously, the vaccine number is generated in a range of which has reduced to 4% of the total capacity. 
        # hence, multiplying by 25 will resume these numbers back to 100% scale.
        for k, v in vacc_sol.items():
            vacc_sol[k] *= 25

        # obtain the name of vaccination centre
        for k,v in ct_name_dict.items():
            if k == ct_capa_list[0]:
                centreName = v

        # get number of a particular vaccination centre type used
        n = ct_num_dict[ct_capa_list[0]]
        # increment total number of each vaccine type 
        total_vacA += (vacc_sol['>60'] * n)
        total_vacB += (vacc_sol['35-60'] * n)
        total_vacC += (vacc_sol['<35'] * n)
     
        table = pd.DataFrame({'Type': ["Centre name", "Vac-A distributed", "Vac-B distributed", "Vac-C distributed"], 
        'Value': [centreName, vacc_sol['>60'], vacc_sol['35-60'], vacc_sol['<35']]})
        table = table.astype(str)
        st.write(table)
        # remove current vaccination centre's capacity from the list
        ct_capa_list.pop(0)

    # obtain the populations of a state
    # vacA - age > 60 
    # vacB - age between 35 to 60
    # vacC - age < 35
    vacA = populations[i][2]
    vacB = populations[i][1]
    vacC = populations[i][0]

    # calculate the days estimated to complete vaccination of each vaccine type
    vacA_day = math.ceil(vacA / total_vacA)
    vacB_day = math.ceil(vacB / total_vacB)
    vacC_day = math.ceil(vacC / total_vacC)
    st.write('Total vaccines distributed per day: ', max_val)
    st.write('Total Vac-A per day: ', total_vacA)
    st.write('Total Vac-B per day: ', total_vacB)
    st.write('Total Vac-C per day: ', total_vacC)
    # choose the maximum number of day required among all vaccine types
    st.write('Number of days to complete vaccination: ', max(vacA_day, vacB_day, vacC_day))
    st.write('________________________________________________________________')
