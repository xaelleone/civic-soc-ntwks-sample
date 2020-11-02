import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
import json
import csv
import editdistance
from census import Census
from us import states
from collections import defaultdict
import editdistance
from fuzzywuzzy import fuzz, process
import math
import scipy.stats as stats

census = Census('001defa3c2cc590165c8b7918f662172c39d3161')
remove_prefixes = ['REV ', 'REVEREND ', 'MR ', 'MS ', 'DR ', 'MRS ', 'FATHER ', 'FR ', 'SISTER ', 'SR ']
# not including ones with commas
remove_suffixes = [' JR', ' SR', ' III']
metric_names = ['density', 'clustering', 'centralization', 'constraint', 'nodes', 'edges', 'pisolate', 'pgiantcomponent', 'BOARD_MEMBER_COUNT', 'INTERLOCKERS']
# total population, white, black, foreign born, income, Gini, bachelor's degree, in state
census_vars = ['B01003_001E', 'B02001_002E', 'B02001_003E', 'B05002_013E', 'B19013_001E', 'B19083_001E', 'B06009_005E', 'B06009_007E', 'B05010_002E']

lower_pop_limit = 20000
upper_pop_limit = 400000
threshold = 90

# turn census name into name that matches with tax documents
def clean_name(c):
    if c.endswith(' city'):
        c = c[:-5]
    if c.endswith(' Town'):
        c = c[:-5]
    return c.upper()

# get a list of cities in census table for one state
def get_city_list (state_fips):
    return list(state_fips[state_fips['Area Name (including legal/statistical area description)'].str.contains('city')]['Area Name (including legal/statistical area description)'].unique())

# load data for all cities in one state
def load_cities (cities, state, state_fips_code, state_fips):
    fname = '/Users/kmli/Development/urop/steil/civil_social_networks/data/output/' + state + '_' + year + '.txt'
    if os.path.isfile(fname):
        with open(fname, 'r') as fin:
            state_orgs = json.load(fin)
    else:
        print('what')
    print(state, len(state_orgs))
    output = defaultdict(list)
    for c in cities:
        acs_metrics = get_acs_metrics(c, state_fips_code, state_fips)
        if not acs_metrics:
            print('census didnt find' + c)
            continue
        pop = acs_metrics[0]['B01003_001E']
        if pop < lower_pop_limit or pop > upper_pop_limit:
            continue
        if state == 'VA' and clean_name(c) == 'ALEXANDRIA':
            continue

        print(clean_name(c), state)
        subtable = [x for x in state_orgs if isinstance(x, dict) and x['city'].upper() == clean_name(c)]
        graph, total_board_members, interlockers = construct_graph(subtable)
        if not graph.edges() or len(graph.edges()) > 10000:
            continue
        network_metrics = get_network_metrics(graph)
        output['name'].append(clean_name(c) + "_" + state)
        network_metrics.append(total_board_members)
        network_metrics.append(interlockers)
        for i, m in enumerate(metric_names):
            output[m].append(network_metrics[i])
        for x in census_vars:
            output[x].append(acs_metrics[0][x])
    return output

# write output from all cities in all states in one year to table
def export_output (output_list):
    with open("awsout/aws_table_" + year + "_isolatestats.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(output_list[0].keys())
        for x in output_list:
            writer.writerows(zip(*x.values()))

# preprocessing for people's names, uses a rule-based approach to filter common name problems
def process_name (name):
    name = str(name).upper()
    name = name.replace('.', '')
    name = name.replace('-', ' ')
    if len(name) <= 4:
        return name
    for pre in remove_prefixes: # if person has a title
        if name[:len(pre)] == pre:
            name = name[len(pre):]
    for suf in remove_suffixes:
        if name[-len(suf):] == suf:
            name = name[:-len(suf)]
    signal_indices = list(map(lambda x: name.find(x), [',', ' (', ')', '\"']))
    if signal_indices[0] > 0: # if person has a degree
        name = name[:signal_indices[0]]
    if signal_indices[1] > 0 and signal_indices[2] > signal_indices[1]: # if person has a nickname in parentheses
        name = name[:signal_indices[1]] + name[signal_indices[2] + 1:]
    return name

# remove board members which are actually banks, which are unusually common
def process_row (row):
    name = str(row['P5NAME'])
    if name != 'nan' and not name.endswith(' BANK'):
        return str(process_name(name))

# take list of board members and construct network model, and return network model
def construct_graph(subtable):
    orgs = dict([(org['ein'], {process_name(name) for name in org['board']}) for org in subtable])
    G = nx.Graph()
    all_board_members = set()
    interlockers = set()
    for x in orgs:
        G.add_node(x)
        all_board_members |= orgs[x]
        for y in orgs:
            if x != y:
                intersect = orgs[x] & orgs[y]
                wt = len(intersect)
                interlockers |= intersect
                if wt != 0:
                    G.add_edge(x, y, weight=wt)
    return G, len(all_board_members), len(interlockers)

# compute freeman centralization of network model
def centralization(G):
    N = G.order()
    degrees = dict(G.degree).values()
    max_deg = max(degrees)
    return float(N*max_deg - sum(degrees))/(N-1)**2

# return the the largest connected component
def giant_component(G):
    return nx.connected_components(G)[0]

# returns a tuple about constraint distribution:
# mean constraint
# nan count
# 1.0 count
# 0.5 count
# gammadist alpha
# gammadist loc
# gammadist beta
def constraint_statistics(constraint_dist):
    nan_count = sum(math.isnan(x) for x in constraint_dist)
    one_count = constraint_dist.count(1)
    half_count = constraint_dist.count(0.5)
    filtered = [x for x in constraint_dist if (not math.isnan(x)) and x != 0.5 and x != 1]
    if len(filtered) != 0:
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(filtered)
    else:
        fit_alpha, fit_loc, fit_beta = (0, 0, 0)
    return nan_count, one_count, half_count, fit_alpha, fit_loc, fit_beta

# returns count of nan constraint, one count, half count, and the four smallest central moments of distribution
def more_constraint_stats(constraint_dist):
    nan_count = sum(math.isnan(x) for x in constraint_dist)
    one_count = constraint_dist.count(1)
    half_count = constraint_dist.count(0.5)
    filtered = [x for x in constraint_dist if (not math.isnan(x)) and x != 0.5 and x != 1]
    return nan_count, one_count, half_count, stats.tmean(filtered), stats.tvar(filtered), stats.skew(filtered), stats.kurtosis(filtered)

# return the proportion of nodes in the largest connected component
def p_giant_component(G):
    return len(giant_component(G).nodes()) / len(G.nodes())

# return a list of relevant statistics about network model
def get_network_metrics(G):
    # print(len(G.nodes()), len(G.edges()))
    constraint = list(nx.constraint(G, weight='weight').values())
    # constraint_stats = more_constraint_stats(constraint)
    # esize = nx.effective_size(G)
    # efficiency = {n: (v / G.degree(n) if G.degree(n) != 0 else 0) for n, v in esize.items()}
    constraint = [x for x in constraint if not math.isnan(x)]
    return [nx.density(G),
        nx.average_clustering(G, weight='weight'),
        centralization(G),
        np.mean(constraint),
        len(G.nodes()),
        len(G.edges()),
        1.0 * len(list(nx.isolates(G))) / len(G.nodes()),
        1.0 * len(max(nx.connected_component_subgraphs(G), key=len)) / len(G.nodes())]

# query census api to get demographic statistics for one city
def get_acs_metrics(c, state_fips_code, state_fips):
    local_fips = state_fips[state_fips['Area Name (including legal/statistical area description)'] == c]
    local_fips = list(local_fips[local_fips['Place Code (FIPS)'] != 0]['Place Code (FIPS)'])
    if not local_fips:
        return []
    local_fips = local_fips[0]
    results = census.acs5.state_place(tuple(census_vars), state_fips_code, local_fips, year=2015)
    return results

# query black-white dissimilarity, a measure of
def get_dissimilarity(c, state):
    di_table = pd.read_csv('bw-dissimilarity/res/' + state + '_DI.csv')
    dis = list(di_table[di_table['CITY'] == c]['BW_DI'])
    if not dis:
        return 0
        print(c)
    return dis[0]

# load board members, construct network models, and export output for each city and state in one year
def main ():
    with open('output_archive/state_abbrs.txt') as fin:
        state_abbrs = [x.strip() for x in fin.readlines()]
    output = []
    for state in state_abbrs:
        print(state)
        state_fips_code = states.lookup(state).fips
        state_fips = pd.read_csv('output_archive/fips_codes.txt', sep='\t', encoding='utf-16')
        state_fips = state_fips[state_fips['State Code (FIPS)'] == int(state_fips_code)]
        output.append(load_cities(get_city_list(state_fips), state, state_fips_code, state_fips))
    export_output(output)

for y in range(2011, 2018):
    year = str(y)
    main()
