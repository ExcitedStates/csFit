"""

    csFit: multi-conformer chemical shift fitting using quadratic programing with constraints
    R. Bryn Fenwick, Henry van den Bedem, David Oyen, Jane Dyson, Peter Wright
    e-mail: wright@scripps.edu 

 
        Copyright (C) 2020 The Scripps Research Institute

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

         http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License. 

"""

import pandas as pd
import cvxpy as cp
import numpy as np
import json
import tqdm
import itertools
from collections import defaultdict
#import smtplib


def send_email(message_text):
    global parameters
    # message
    _from = parameters['email_sending_address']
    _to = parameters['email_receiving_address']
    message = "From: {}\nTo: {}\nSubject: csFit\n\n{}".format(_from, _to, message_text)

    # creates SMTP session 
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security 
    s.starttls()
    # Authentication 
    s.login(_from, parameters['email_sending_password'])
    # sending the mail 
    s.sendmail(_from, _to, message)
    # terminating the session 
    s.quit()


def run_enumeration(A, b, A_add, b_add, threshold=0.1, n=5, keep=10):
    score_array = []

    # max_columns: ensemble members
    # max_rows: data points
    A_all = np.concatenate((A, A_add))
    max_rows, max_columns = A_all.shape
    print(' ensemble members: {}'.format(max_columns))

    # for the sign enumeration
    signs = np.array([-1, 1])
    sign_set = [signs * i for i in b]
    b_sign = itertools.product(*sign_set)

    # for the solver
    # Construct a CVXPY problem
    # weights for the population members
    w = cp.Variable(max_columns)
    # indicator of if the member is used
    b_all = cp.Parameter(max_rows)
    indicator = cp.Variable(max_columns, boolean=True)
    constraints = [threshold * indicator - w <= 0,
                   w - indicator <= 0,
                   sum(w) == 1.0,
                   sum(indicator) <= n]

    # cost function for the problem
    cost = cp.sum_squares(A_all @ w - b_all)
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)

    total = 2 ** (len(b))
    print(' sign evaluations: {}'.format(total))
    for b in tqdm.tqdm(b_sign, total=total, mininterval=10, maxinterval=100):
        #
        b_all.value = np.concatenate((b, b_add))

        # solve the problem
        prob.solve(solver=cp.CPLEX, warm_start=True)

        new_value = float(prob.value)
        if len(score_array) < keep:
            score_array.append({'objective': new_value, 'weight': w.value, 'indicator': indicator.value})
            score_array = sorted(score_array, key=lambda k: k['objective'], reverse=False)
        elif new_value <= score_array[-1]['objective']:
            score_array.append({'objective': new_value, 'weight': w.value, 'indicator': indicator.value})
            # and let the last one go...
            score_array = sorted(score_array, key=lambda k: k['objective'], reverse=False)[:-1]
        else:
            pass

#    try:
#        send_email('calculations complete!')
#    except smtplib.SMTPAuthenticationError:
#        pass
    return score_array


def read_selection(file_name):
    data_dict = []
    with open(file_name) as fh:
        for line in fh:
            if line[0] == '#':
                # because we will not enumerate the signs we eliminate the delta as well
                unsigned = 1
                delta = 0.0
                resi, atom, _, weight = line.strip()[1:].split()
                pass
            else:
                unsigned = 0
                resi, atom, delta, weight = line.strip().split()
            data_dict.append({'resi': resi, 'atom': atom, 'delta': float(delta), 'weight': float(weight),
                              'unsigned': bool(unsigned)})
    return pd.DataFrame(data_dict)


def read_ensemble(file_name):
    data_dict = defaultdict(lambda: defaultdict())
    with open(file_name) as input_fh:
        for line in input_fh:
            file_h = line.strip()
            with open(file_h) as fh:
                structure_number = (file_h.split('/')[-1].split('.')[0])
                for file_line in fh:
                    if 'resi' in file_line:
                        pass
                    else:
                        data = file_line.strip().split()
                        resi = data[0]
                        atom = data[1]
                        sparta_shift = float(data[8])
                        reference_shift = float(data[5])
                        data_dict[structure_number]["{}_{}".format(resi, atom)] = (sparta_shift - reference_shift)
    df = pd.DataFrame(data_dict).reset_index()
    # Adding two new columns to the existing data frame.
    # by default splitting is done on the basis of single space.
    df[['resi', 'atom']] = df['index'].str.split('_', expand=True)
    df.drop(columns=['index'], inplace=True)
    return df


if __name__ == "__main__":

    from sys import argv

    input_file = argv[1]
    with open(input_file) as json_file:
        parameters = json.load(json_file)[0]
    print(parameters)

#    try:
#        send_email('check if email alerts are working!')
#    except smtplib.SMTPAuthenticationError:
#        print('Warning: email notifications are not working')

    b_df = read_selection(parameters['selection'])
    A_df = read_ensemble(parameters['ensemble'])
    full_df = pd.merge(b_df, A_df, how='inner', on=['atom', 'resi'])

    # prepare the part for sign enumeration
    signed_df = full_df.drop(full_df[full_df['unsigned'] == 1].index)
    signed_weight = signed_df.weight.to_numpy()
    signed_delta = signed_df.delta.to_numpy()

    signed_df = signed_df.drop(columns=['resi', 'atom', 'delta', 'weight', 'unsigned'])
    signed_A = signed_df.to_numpy().T * signed_weight.T
    signed_b = signed_delta * signed_weight

    # prepare the part that is constant
    unsigned_df = full_df.drop(full_df[full_df['unsigned'] == 0].index)
    unsigned_weight = unsigned_df.weight.to_numpy()
    unsigned_delta = unsigned_df.delta.to_numpy()

    unsigned_df = unsigned_df.drop(columns=['resi', 'atom', 'delta', 'weight', 'unsigned'])
    unsigned_A = unsigned_df.to_numpy().T * unsigned_weight.T
    unsigned_b = unsigned_delta * unsigned_weight

    # run enumeration
    result_array = run_enumeration(signed_A.T, signed_b.T, unsigned_A.T, unsigned_b.T,
                                   threshold=parameters['threshold'], n=parameters['ensemble_size'],
                                   keep=parameters['output_ensembles'])

    with np.printoptions(precision=3, suppress=True):
        for result in range(parameters['output_ensembles']):
            with open('result_{k}.out'.format(k=result), 'w') as output_fh:
                result_objective = result_array[result]['objective']
                result_weight = result_array[result]['weight']
                result_indicator = result_array[result]['indicator']
                output_fh.write('# max_structures {ensemble_size}\n'.format(ensemble_size=parameters['ensemble_size']))
                output_fh.write('# threshold_value {threshold}\n'.format(threshold=parameters['threshold']))
                output_fh.write('# objective_value {objective}\n'.format(objective=result_objective))
                output_fh.write('population_fraction structure\n'.format(objective=result_objective))
                for structure, value in enumerate(result_indicator):
                    if value >= 0.9:  # deal with any potential rounding
                        output_fh.write('{value:9.7f} {structure}\n'.format(value=result_weight[structure],
                                                                            structure=signed_df.columns[structure]))
