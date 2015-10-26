"""
Functions for manipulating epanet engine and inp. files.
"""
import datetime
import os
import re
from os.path import join

import networkx as nx
from numpy import mean
from pandas import DataFrame

from epanettools import epanet2 as et
from pyEpanet.utils import find_edges, set_middle_node

__author__ = 'Alexey Karnachev'


def run_hydraulic_simulation(inp_file, nodes_parameters=None, links_parameters=None):
    """
    Function for simulate hydraulic model from inp. file.
    @param inp_file: path to inp. file.
    @param nodes_parameters: list of nodes parameters to simulate.
    @param links_parameters: list of links parameters to simulate.
    @return: dict with simulated data.
    """

    __NODE_TYPES = {'JUNCTION': 0, 'RESERVOIR': 1, 'TANK': 2}

    __NODES_VALUES = {'ELEVATION': 0, 'BASEDEMAND': 1, 'PATTERN': 2, 'EMITTER': 3, 'INITQUAL': 4,
                      'SOURCEQUAL': 5, 'SOURCEPAT': 6, 'SOURCETYPE': 7, 'TANKLEVEL': 8, 'DEMAND': 9,
                      'HEAD': 10, 'PRESSURE': 11, 'QUALITY': 12, 'SOURCEMASS': 13}

    __LINKS_VALUES = {'DIAMETER': 0, 'LENGTH': 1, 'ROUGHNESS': 2, 'MINORLOSS': 3, 'INITSTATUS': 4,
                      'INITSETTING': 5, 'KBULK': 6, 'KWALL': 7, 'FLOW': 8, 'VELOCITY': 9,
                      'HEADLOSS': 10, 'STATUS': 11, 'SETTING': 12, 'ENERGY': 13}

    if nodes_parameters is None:
        nodes_parameters = []
    if links_parameters is None:
        links_parameters = []

    # Prepare some paths:
    rpt_file = re.sub('inp$', 'rpt', inp_file)
    out_file = re.sub('inp$', 'out', inp_file)

    err = et.ENopen(inp_file, rpt_file, out_file)
    if err != 0:
        raise ValueError(et.ENgeterror(err, 256)[1])

    # Create empty dict. for data:
    data = {'NODES': {'DEMAND': {},
                      'HEAD': {},
                      'PRESSURE': {},
                      'QUALITY': {},
                      'SOURCEMASS': {}},
            'LINKS': {'FLOW': {},
                      'VELOCITY': {},
                      'HEADLOSS': {},
                      'STATUS': {},
                      'SETTING': {}}}

    # Run hydraulic modeling:
    et.ENopenH()
    et.ENinitH(0)

    n_nodes = et.ENgetcount(0)[1]
    n_links = et.ENgetcount(2)[1]

    not_finished = True
    while not_finished:
        et.ENrunH()
        for parameter in nodes_parameters:
            parameter_ind = __NODES_VALUES[parameter]
            for i in range(1, n_nodes + 1):
                id = et.ENgetnodeid(i)[1]
                if id not in data['NODES'][parameter]:
                    data['NODES'][parameter][id] = [et.ENgetnodevalue(i, parameter_ind)[1]]
                else:
                    data['NODES'][parameter][id] += [et.ENgetnodevalue(i, parameter_ind)[1]]
        for parameter in links_parameters:
            parameter_ind = __LINKS_VALUES[parameter]
            for i in range(1, n_links + 1):
                id = et.ENgetlinkid(i)[1]
                if id not in data['LINKS'][parameter]:
                    data['LINKS'][parameter][id] = [et.ENgetlinkvalue(i, parameter_ind)[1]]
                else:
                    data['LINKS'][parameter][id] += [et.ENgetlinkvalue(i, parameter_ind)[1]]
        tstep = et.ENnextH()[1]
        if tstep <= 0:
            not_finished = False

    # Close Epanet
    et.ENclose()
    os.remove(rpt_file)
    return data


def simulate_head_data(inp_file):
    """
    Function to simulate head data.
    @param inp_file: path to inp. file.
    @return: pandas DataFrame with simulated head data.
    """

    return DataFrame(run_hydraulic_simulation(inp_file, ['HEAD'])['NODES']['HEAD'])


def inp_to_graph(inp_file):
    """
    Function to import inp. file into directed networkx graph structure.
    @param inp_file: path to inp. file.
    @return: directed networkx graph object.
    """
    inp_file = inp_file
    # ==================================================================================================================
    # Initial Module:

    # Graph initialization
    G = nx.DiGraph()
    G.graph['TITLE'] = ''
    G.graph['PATTERNS'] = {}
    G.graph['CURVES'] = {}
    G.graph['CONTROLS'] = ''
    G.graph['RULES'] = ''
    G.graph['ENERGY'] = ''
    G.graph['REACTIONS'] = ''
    G.graph['TIMES'] = {}
    G.graph['REPORT'] = ''
    G.graph['OPTIONS'] = {}
    G.graph['COORDINATES'] = {}
    G.graph['VERTICES'] = {}
    G.graph['LABELS'] = {}
    G.graph['BACKDROP'] = ''

    times_parameters = ['duration', 'hydraulic timestep', 'quality timestep', 'pattern timestep', 'pattern start',
                        'report timestep', 'report start', 'start clocktime', 'statistic']

    options_parameters = ['units', 'headloss', 'hydraulic', 'quality', 'viscosity', 'diffusivity', 'specific gravity',
                          'trials', 'accuracy', 'unbalanced', 'pattern', 'demand multiplier', 'emitter exponent',
                          'tolerance', 'map']

    # Each line processing
    section = ''
    with open(inp_file) as f:
        for line in f:

            # ==========================================================================================================
            # Preparing Module:

            # Get section name from line
            if re.match('\[([A-Z]+)\]', line):
                section = re.sub('\W', '', line)
                continue

            # Pass an empty line and a comment line
            if len(line.strip()) == 0 or line.strip()[0] == ';':
                continue

            # Remove comments from line
            line = re.sub(';.*', '', line)

            # ==========================================================================================================
            # Processing Module:

            # [TITLE]:
            if section == 'TITLE':
                G.graph['TITLE'] += line

            # [JUNCTIONS]:
            if section == 'JUNCTIONS':
                row = line.split()
                type = 'JUNCTION'
                node = row[0]
                elevation = float(row[1])
                basedemand = 0.0
                pattern = ''
                if len(row) > 2:
                    basedemand = float(row[2])
                if len(row) > 3:
                    pattern = row[3]
                G.add_node(n=node,
                           attr_dict={'ELEVATION': elevation,
                                      'BASEDEMAND': basedemand,
                                      'PATTERN': pattern,
                                      'TYPE': type})

            # [RESERVOIRS]:
            if section == 'RESERVOIRS':
                row = line.split()
                type = 'RESERVOIR'
                node = row[0]
                head = float(row[1])
                pattern = ''
                if len(row) > 2:
                    pattern = row[2]
                G.add_node(n=node,
                           attr_dict={'HEAD': head,
                                      'PATTERN': pattern,
                                      'TYPE': type})

            # [TANKS]:
            if section == 'TANKS':
                row = line.split()
                type = 'TANK'
                node = row[0]
                elev = float(row[1])
                initlvl = float(row[2])
                minlvl = float(row[3])
                maxlvl = float(row[4])
                diam = float(row[5])
                minvol = float(row[6])
                pattern = ''
                if len(row) > 7:
                    pattern = float(row[7])
                G.add_node(n=node,
                           attr_dict={'ELEVATION': elev,
                                      'INIT_LEVEL': initlvl,
                                      'MIN_LEVEL': minlvl,
                                      'MAX_LEVEL': maxlvl,
                                      'DIAMETER': diam,
                                      'MIN_VOLUME': minvol,
                                      'CURVE': pattern,
                                      'TYPE': type})

            # [PIPES]:
            if section == 'PIPES':
                row = line.split()
                type = 'PIPE'
                link = row[0]
                fromnode = row[1]
                tonode = row[2]
                length = float(row[3])
                diam = float(row[4])
                roughness = float(row[5])
                minloss = float(row[6])
                status = row[7]
                G.add_edge(fromnode, tonode,
                           attr_dict={'LENGTH': length,
                                      'DIAMETER': diam,
                                      'ROUGHNESS': roughness,
                                      'MINOR_LOSS': minloss,
                                      'STATUS': status,
                                      'ID': link,
                                      'TYPE': type})

            # [PUMPS]:
            if section == 'PUMPS':
                row = line.split()
                type = 'PUMP'
                link = row[0]
                fromnode = row[1]
                tonode = row[2]
                params = ' '.join(row[3:])
                G.add_edge(fromnode, tonode,
                           attr_dict={'PARAMETERS': params,
                                      'ID': link,
                                      'TYPE': type})

            # [VALVES]:
            if section == 'VALVES':
                row = line.split()
                type = row[4]
                link = row[0]
                fromnode = row[1]
                tonode = row[2]
                diam = float(row[3])
                setting = row[5]
                minloss = row[6]
                G.add_edge(fromnode, tonode,
                           attr_dict={'DIAMETER': diam,
                                      'ID': link,
                                      'TYPE': type,
                                      'SETTING': setting,
                                      'MINOR_LOSS': minloss})

            # [TAGS]:
            if section == 'TAGS':
                row = line.split()
                tag = row[2]
                if row[0].lower() == 'node':
                    node = row[1]
                    G.node[node]['TAG'] = tag
                if row[0].lower() == 'link':
                    link_id = row[1]
                    for n1, n2 in G.edges():
                        if G.edge[n1][n2]['ID'] == link_id:
                            G.edge[n1][n2]['TAG'] = tag

            # [DEMANDS]:
            if section == 'DEMANDS':
                pattern = ''
                row = line.split()
                node = row[0]
                basedemand = float(row[1])
                if len(row) == 3:
                    pattern = row[2]
                G.node[node]['PATTERN'] = pattern
                G.node[node]['BASEDEMAND'] = basedemand

            # [STATUS]:
            if section == 'STATUS':
                row = line.split()
                link_id = row[0]
                status = row[1]
                for n1, n2 in G.edges():
                    if G.edge[n1][n2]['ID'] == link_id:
                        G.edge[n1][n2]['INIT_STATUS'] = status


            # [PATTERNS]:
            if section == 'PATTERNS':
                row = line.split()
                pattern = row[0]
                values = [float(i) for i in row[1:]]
                if pattern not in G.graph['PATTERNS']:
                    G.graph['PATTERNS'][pattern] = values
                else:
                    G.graph['PATTERNS'][pattern] += values

            # [CURVES]:
            if section == 'CURVES':
                row = line.split()
                pattern = row[0]
                values = [tuple(float(i) for i in row[1:])]
                if pattern not in G.graph['CURVES']:
                    G.graph['CURVES'][pattern] = values
                else:
                    G.graph['CURVES'][pattern] += values

            # [CONTROLS]:
            if section == 'CONTROLS':
                G.graph['CONTROLS'] += line

            # [RULES]:
            if section == 'RULES':
                G.graph['RULES'] += line

            # [ENERGY]:
            if section == 'ENERGY':
                G.graph['ENERGY'] += line

            # [EMITTERS]:
            if section == 'EMITTERS':
                row = line.split()
                node = row[0]
                emitter = float(row[1])
                G.node[node]['EMITTER'] = emitter

            # [QUALITY]:
            if section == 'QUALITY':
                row = line.split()
                node = row[0]
                quality = float(row[1])
                G.node[node]['QUALITY'] = quality

            # [SOURCES]:
            if section == 'SOURCES':
                row = line.split()
                node = row[0]
                type = row[1]
                strength = float(row[2])
                if len(row) == 4:
                    pattern = row[3]
                    G.node[node]['PATTERN'] = pattern
                G.node[node]['SOURCE_TYPE'] = type
                G.node[node]['SOURCE_STRENGTH'] = strength

            # [REACTIONS]:
            if section == 'REACTIONS':
                G.graph['REACTIONS'] += line

            # [MIXING]:
            if section == 'MIXING':
                row = line.split()
                node = row[0]
                model = ' '.join(row[1:])
                G.node[node]['MIXING_MODEL'] = model

            # [TIMES]:
            if section == 'TIMES':
                line = line.lower()
                for times_parameter in times_parameters:
                    if re.findall(times_parameter, line):
                        value = re.sub(times_parameter, '', line).strip()
                        G.graph['TIMES'][times_parameter] = value

            # [REPORT]:
            if section == 'REPORT':
                G.graph['REPORT'] += line

            # [OPTIONS]:
            if section == 'OPTIONS':
                for options_parameter in options_parameters:
                    if re.search(options_parameter, line, re.IGNORECASE):
                        value = re.sub(options_parameter, '', line, flags=re.IGNORECASE).strip()
                        G.graph['OPTIONS'][options_parameter] = value

            # [COORDINATES]:
            if section == 'COORDINATES':
                row = line.split()
                node = row[0]
                x = float(re.sub(',', '.', row[1]))
                y = float(re.sub(',', '.', row[2]))
                G.graph['COORDINATES'][node] = (x, y)

            # [VERTICES]:
            if section == 'VERTICES':
                row = line.split()
                pattern = row[0]
                values = [tuple(float(re.sub(',', '.', i)) for i in row[1:])]
                if pattern not in G.graph['VERTICES']:
                    G.graph['VERTICES'][pattern] = values
                else:
                    G.graph['VERTICES'][pattern] += values

            # [LABELS]:
            if section == 'LABELS':
                row = line.split()
                node = row[2]
                x = float(row[0])
                y = float(row[1])
                G.graph['LABELS'][node] = (x, y)

            # [BACKDROP]:
            if section == 'BACKDROP':
                G.graph['BACKDROP'] += line

    return G


def graph_to_inp(inp_file, G, ignore_section=None):
    """
    Function to export networkx graph into inp. file.
    @param inp_file: path to inp. file.
    @param ignore_section: sections of the inp. file, which will not be written.
    @param G: networkx graph object.
    @return: write new inp. file.
    """
    if ignore_section is None:
        ignore_section = []

    with open(inp_file, 'w') as f:
        # Write nodes in tmp strings:
        junctions = ''
        reservoirs = ''
        tanks = ''
        demands = ''
        tags = ''
        emitters = ''
        quality = ''
        sources = ''
        mixing = ''
        for n in G.nodes_iter():
            node = G.node[n]
            if node['TYPE'] == 'JUNCTION':
                junctions += (' ' * 10).join([n, str(node['ELEVATION']), '\n'])
                pattern = node['PATTERN']
                demand = node['BASEDEMAND']
                demands += (' ' * 10).join([n, str(demand), str(pattern), '\n'])

            elif node['TYPE'] == 'RESERVOIR':
                reservoirs += (' ' * 10).join([n, str(node['HEAD']), str(node['PATTERN']), '\n'])

            elif node['TYPE'] == 'TANK':
                tanks += (' ' * 10).join([n, str(node['ELEVATION']), str(node['INIT_LEVEL']),
                                          str(node['MIN_LEVEL']), str(node['MAX_LEVEL']),
                                          str(node['DIAMETER']), str(node['MIN_VOLUME']),
                                          str(node['CURVE']), '\n'])
            if 'TAG' in node:
                tags += (' ' * 10).join(['NODE', n, str(node['TAG']), '\n'])

            if 'EMITTER' in node:
                emitters += (' ' * 10).join([n, str(node['EMITTER']), '\n'])

            if 'QUALITY' in node:
                quality += (' ' * 10).join([n, str(node['QUALITY']), '\n'])

            if 'SOURCE_TYPE' in node:
                sources += (' ' * 10).join([n, str(node['SOURCE_TYPE']), str(node['SOURCE_STRENGTH']),
                                            str(node['PATTERN']), '\n'])

            if 'MIXING_MODEL' in node:
                mixing += (' ' * 10).join([n, str(node['MIXING_MODEL']), '\n'])

        # Write edges in tmp strings:
        pipes = ''
        pumps = ''
        valves = ''
        status = ''
        for n1, n2 in G.edges_iter():
            edge = G.edge[n1][n2]
            if edge['TYPE'] == 'PIPE':
                pipes += (' ' * 10).join([edge['ID'], n1, n2,
                                          str(edge['LENGTH']), str(edge['DIAMETER']),
                                          str(edge['ROUGHNESS']), str(edge['MINOR_LOSS']),
                                          str(edge['STATUS']), '\n'])
            if edge['TYPE'] == 'PUMP':
                pumps += (' ' * 10).join([edge['ID'], n1, n2, str(edge['PARAMETERS']), '\n'])

            if edge['TYPE'] == 'VALVE':
                pumps += (' ' * 10).join([edge['ID'], n1, n2, str(edge['DIAMETER']),
                                          str(edge['TYPE']), str(edge['SETTING']),
                                          str(edge['MINOR_LOSS']), '\n'])

            if 'TAG' in edge:
                tags += (' ' * 10).join(['LINK', edge['ID'], str(edge['TAG']), '\n'])

            if 'INIT_STATUS' in edge:
                status += (' ' * 10).join([edge['ID'], str(edge['INIT_STATUS']), '\n'])

        # Write patterns in tmp string:
        patterns = ''
        for p in G.graph['PATTERNS']:
            pattern = G.graph['PATTERNS'][p]
            values = []
            for value in pattern:
                values.append(value)
                if len(values) == 6:
                    values = [str(i) for i in values]
                    patterns += p + (' ' * 10) + (' ' * 10).join(values) + '\n'
                    values = []

        # Write times in tmp string
        times = ''
        for t in G.graph['TIMES']:
            value = G.graph['TIMES'][t]
            times += t + (' ' * 10) + value + '\n'

        # Write options in tmp string
        options = ''
        for o in G.graph['OPTIONS']:
            value = G.graph['OPTIONS'][o]
            options += o + (' ' * 10) + str(value) + '\n'

        # Write curves in tmp string:
        curves = ''
        for c in G.graph['CURVES']:
            curve = G.graph['CURVES'][c]
            for values in curve:
                values = [str(i) for i in values]
                curves += c + (' ' * 10) + (' ' * 10).join(values) + '\n'

        # Write coordinates in tmp string:
        coordinates = ''
        for c in G.graph['COORDINATES']:
            coord = G.graph['COORDINATES'][c]
            coord = [str(i) for i in coord]
            coordinates += c + (' ' * 10) + (' ' * 10).join(coord) + '\n'

        # Write vertices in tmp string:
        vertices = ''
        for v in G.graph['VERTICES']:
            vertex = G.graph['VERTICES'][v]
            for coord in vertex:
                coord = [str(i) for i in coord]
                vertices += v + (' ' * 10) + (' ' * 10).join(coord) + '\n'

        # Write coordinates in tmp string:
        labels = ''
        for c in G.graph['LABELS']:
            labels += (' ' * 10).join([str(G.graph['LABELS'][c][0]),
                                       str(G.graph['LABELS'][c][1]),
                                       c, '\n'])

        section = 'TITLE'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(G.graph[section])
            f.write('\n')

        section = 'JUNCTIONS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(junctions)
            f.write('\n')

        section = 'RESERVOIRS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(reservoirs)
            f.write('\n')

        section = 'TANKS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(tanks)
            f.write('\n')

        section = 'PIPES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(pipes)
            f.write('\n')

        section = 'PUMPS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(pumps)
            f.write('\n')

        section = 'VALVES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(valves)
            f.write('\n')

        section = 'TAGS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(tags)
            f.write('\n')

        section = 'DEMANDS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(demands)
            f.write('\n')

        section = 'STATUS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(status)
            f.write('\n')

        section = 'PATTERNS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(patterns)
            f.write('\n')

        section = 'CURVES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(curves)
            f.write('\n')

        section = 'CONTROLS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(G.graph[section])
            f.write('\n')

        section = 'RULES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(G.graph[section])
            f.write('\n')

        section = 'ENERGY'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(G.graph[section])
            f.write('\n')

        section = 'EMITTERS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(emitters)
            f.write('\n')

        section = 'QUALITY'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(quality)
            f.write('\n')

        section = 'SOURCES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(sources)
            f.write('\n')

        section = 'REACTIONS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(G.graph[section])
            f.write('\n')

        section = 'MIXING'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(mixing)
            f.write('\n')

        section = 'TIMES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(times)
            f.write('\n')

        section = 'REPORT'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(G.graph[section])
            f.write('\n')

        section = 'OPTIONS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(options)
            f.write('\n')

        section = 'COORDINATES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(coordinates)
            f.write('\n')

        section = 'VERTICES'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(vertices)
            f.write('\n')

        section = 'LABELS'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(labels)
            f.write('\n')

        section = 'BACKDROP'
        if section not in ignore_section:
            f.write('[{}]\n'.format(section))
            f.write(G.graph[section])
            f.write('\n')


def prepare_leakages_scenarios(basic_inp_file, leaks_demands, output_folder):
    """
    Function to prepare leakages scenarios on all pipes.
    @param basic_inp_file: path to reference inp. file.
    @param leaks_demands: list of the leak demands values.
    @param output_folder: folder with output scenarios inp. files
    """

    # Import basic inp. file:
    g = inp_to_graph(basic_inp_file)

    # Simulate basic scenario:
    noleak_data = run_hydraulic_simulation(basic_inp_file, nodes_parameters=['PRESSURE'])
    noleak_df = DataFrame(noleak_data['NODES']['PRESSURE'])

    # Go through all leak demands:
    edges_list = find_edges(g, lambda x: (x['TYPE'] == 'PIPE'))

    for LEAK_DEMAND in leaks_demands:
        e = 0
        # Go through all edges:
        for edge in edges_list:
            e += 1
            g = inp_to_graph(basic_inp_file)
            node_1 = g.node[edge[0]]
            node_2 = g.node[edge[1]]

            # Set middle point:
            if 'ELEVATION' not in node_1:
                elev_1 = 0
            else:
                elev_1 = node_1['ELEVATION']

            if 'ELEVATION' not in node_2:
                elev_2 = 0
            else:
                elev_2 = node_2['ELEVATION']

            new_elevation = (elev_1 + elev_2) / 2

            middle_node_dict = {'TYPE': 'JUNCTION', 'ELEVATION': new_elevation, 'BASEDEMAND': 0.0, 'PATTERN': '',
                                'TAG': 'middle_node'}
            set_middle_node(g, middle_node_dict, edge,
                            {'LENGTH': lambda x: x / 2, 'ID': lambda x: x + '_1'},
                            {'LENGTH': lambda x: x / 2, 'ID': lambda x: x + '_2'})

            # Set new coordinate:
            node_1_coord = g.graph['COORDINATES'][edge[0]]
            node_2_coord = g.graph['COORDINATES'][edge[1]]
            new_x = (node_1_coord[0] + node_2_coord[0]) / 2
            new_y = (node_1_coord[1] + node_2_coord[1]) / 2
            new_node_name = '{node_1}_{node_2}'.format(node_1=edge[0], node_2=edge[1])
            g.graph['COORDINATES'][new_node_name] = (new_x, new_y)

            # Estimate emitter coefficient:
            pressure = (noleak_df[edge[0]] + noleak_df[edge[0]]) / 2
            emitter = LEAK_DEMAND / mean([pow(i, 0.5) for i in pressure])
            g.node[new_node_name]['EMITTER'] = emitter

            # Write scenario meta information:
            g.graph['TITLE'] = 'leak_node = {}\nleak_demand = {}'.format(new_node_name, LEAK_DEMAND)

            # Save tmp scenario inp file:
            new_inp_file_path = output_folder + 'node_{}_leak_{}.inp'.format(edge[0] + '_' + edge[1], LEAK_DEMAND)
            graph_to_inp(new_inp_file_path, g)


def simulate_learn_scenarios_data(inp_files_folder, output_data_file=None):
    """
    Function to simulate head learn data for machine learning tasks.
    @param inp_files_folder: path to folder with inp. scenarios.
    @param output_data_file: path to output data file.
    @return: pandas DataFrame (if output_data_file is None) else return nothing and save data in output_data_file
    """
    # files = [join(inp_files_folder, file) for file in os.listdir(inp_files_folder)]
    files = os.listdir(inp_files_folder)
    learn_data = DataFrame()
    f = 0
    F = len(files)
    for file in files:
        f += 1
        t = datetime.datetime.now().__str__()
        if re.match('^.+\.inp$', file):
            file_path = join(inp_files_folder, file)
            title = inp_to_graph(file_path).graph['TITLE']

            leak_node = re.findall(re.compile('^leak_node = (.+)\n'), title)
            leak_demand = re.findall(re.compile('leak_demand = (.+)$'), title)

            if len(leak_node) != len(leak_demand) != 1:
                print('Can not parse [TITLE] section in .inp file: "{}"'.format(file))
                continue
            else:
                leak_node = leak_node[0]
                leak_demand = leak_demand[0]

            try:
                data = simulate_head_data(file_path)
                data['LEAK_NODE'] = leak_node
                data['LEAK_DEMAND'] = leak_demand
                data = data.drop(leak_node, 1)
                learn_data = learn_data.append(data)
            except ValueError:
                print('Can not simulate: "{}"'.format(file))

            print(t, 'Scenarios processed: {} from {}'.format(f, F))

    if output_data_file is None:
        return learn_data
    else:
        learn_data.to_csv(output_data_file)
