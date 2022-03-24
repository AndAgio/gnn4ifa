# Python modules
import pandas as pd
import numpy as np
import math
import os
import glob
import warnings
import networkx as nx
import torch
import torch_geometric as tg
import matplotlib.pyplot as plt
# Import modules
from gnn4ifa.utils import timeit, get_scenario_labels_dict

warnings.filterwarnings("ignore")


class Extractor():
    def __init__(self, data_dir='ifa_data',
                 scenario='existing',
                 topology='small',
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=300,
                 time_att_start=50):
        self.data_dir = data_dir

        self.scenario = scenario
        self.topology = topology

        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids

        self.simulation_time = simulation_time
        self.time_att_start = time_att_start

    @staticmethod
    def get_data(files):
        # Define empty dictionary containing data
        data = {}
        # Iterate over all simulation files and read them
        for file in files:
            # Check file type
            file_type = file.split('/')[-1].split('-')[0]
            if file_type == 'format':
                continue
            if file_type == 'pit':
                file = Extractor.convert_pit_to_decent_format(file)
            # Read csv file
            file_data = pd.read_csv(file, sep='\t', index_col=False)
            # Put data in dictionary
            data[file_type] = file_data
        return data

    @staticmethod
    def convert_pit_to_decent_format(file):
        # Get number of lines of file
        num_lines = sum(1 for _ in open(file))
        # Get index of lines containing time
        times_lines = [i for i, line in enumerate(open(file)) if 'Simulation time' in line]
        # Get number of routers
        n_routers = times_lines[1] - 1
        # Get lines containing actual data
        data_lines = [line for line in open(file) if 'Simulation time' not in line]
        # Append timing to each line of the data_lines
        final_lines = []
        for i, line in enumerate(open(file)):
            if 'Simulation time' not in line:
                time = i // (n_routers + 1) + 1
                final_lines.append(str(time) + '\t' + line.replace(' ', '\t'))
        # Store file with new name
        new_file = os.path.join(*file.split('/')[:-1])
        new_file = os.path.join('/', new_file, 'format-{}'.format(file.split('/')[-1]))
        if os.path.exists(new_file):
            os.remove(new_file)
        with open(new_file, 'w') as f:
            f.write('Time\tNode\tWord1\tWord2\tWord3\tSize\n')
            for item in final_lines:
                f.write(item)
        return new_file

    @staticmethod
    def get_router_names(data):
        # Get names of transmitter devices
        routers_names = data['rate']['Node'].unique()
        # Consider routers only
        routers_names = [i for i in routers_names if 'Rout' in i]
        # print('routers_names: {}'.format(routers_names))
        return routers_names

    @staticmethod
    def filter_data_by_time(data, time, verbose=False):
        if verbose:
            print('Time: {}'.format(time))
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = data[key][data[key]['Time'] == time]
        return filtered_data

    @staticmethod
    def extract_data_up_to(data, time, verbose=False):
        if verbose:
            print('Time: {}'.format(time))
        filtered_data = {}
        for key, value in data.items():
            filtered_data[key] = data[key][data[key]['Time'] <= time]
        return filtered_data

    def get_graph_structure(self, debug=False):
        # Open file containing train_topology structure
        topology_file = os.path.join(self.data_dir, 'topologies', '{}_topology.txt'.format(self.topology))
        router_links = []
        for line in open(topology_file, 'r'):
            source = line.split(',')[0]
            dest = line.split(',')[1].split('\n')[0]
            if source[:4] == 'Rout' and dest[:4] == 'Rout':
                router_links.append([source[4:], dest[4:]])
        # print('router_links: {}'.format(router_links))
        list_of_nodes = list(set([elem for link in router_links for elem in link]))
        # Use nx to obtain the graph corresponding to the graph
        graph = nx.DiGraph()
        # Build the DODAG graph from nodes and edges lists
        graph.add_nodes_from(list_of_nodes)
        graph.add_edges_from(router_links)
        # Print and plot for debugging purposes
        if debug:
            print('graph: {}'.format(graph))
            print('graph.nodes: {}'.format(graph.nodes))
            print('graph.edges: {}'.format(graph.edges))
            subax1 = plt.subplot(111)
            nx.draw(graph, with_labels=True, font_weight='bold')
            plt.show()
        # Return networkx graph
        return graph

    @staticmethod
    def get_node_features(data, node_name, mode='array'):
        if mode == 'array':
            features = np.zeros((12), dtype=float)
        elif mode == 'dict':
            features = {}
        else:
            raise ValueError('Invalid mode for extracting node features!')
        # Get different modes of data
        rate_data = data['rate']
        pit_data = data['pit']
        drop_data = data['drop']
        # Get pit size of router at hand
        router_index = node_name.split('Rout')[-1]
        pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
        features[0 if mode == 'array' else 'pit_size'] = pit_size
        # Get drop rate of router at hand
        try:
            drop_rate = drop_data[drop_data['Node'] == node_name]['PacketsRaw'].item()
        except ValueError:
            drop_rate = 0
        # if math.isnan(drop_rate):
        #     print('drop rate: {}'.format(drop_rate))
        #     raise ValueError('NaN found in drop rate!')
        features[1 if mode == 'array' else 'drop_rate'] = drop_rate
        # Get InInterests of router at hand
        in_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InInterests')]['PacketRaw']
        in_interests_list = in_interests.to_list()
        in_interests = sum(i for i in in_interests_list)
        features[2 if mode == 'array' else 'in_interests'] = in_interests
        # Get OutInterests of router at hand
        out_interests = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutInterests')]['PacketRaw']
        out_interests_list = out_interests.to_list()
        out_interests = sum(i for i in out_interests_list)
        features[3 if mode == 'array' else 'out_interests'] = out_interests
        # Get InData of router at hand
        in_data = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InData')]['PacketRaw']
        in_data_list = in_data.to_list()
        in_data = sum(i for i in in_data)
        features[4 if mode == 'array' else 'in_data'] = in_data
        # Get OutData of router at hand
        out_data = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutData')]['PacketRaw']
        out_data_list = out_data.to_list()
        out_data = sum(i for i in out_data_list)
        features[5 if mode == 'array' else 'out_data'] = out_data
        # Get InNacks of router at hand
        in_nacks = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InNacks')]['PacketRaw']
        in_nacks_list = in_nacks.to_list()
        in_nacks = sum(i for i in in_nacks_list)
        features[6 if mode == 'array' else 'in_nacks'] = in_nacks
        # Get OutNacks of router at hand
        out_nacks = rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutNacks')]['PacketRaw']
        out_nacks_list = out_nacks.to_list()
        out_nacks = sum(i for i in out_nacks_list)
        features[7 if mode == 'array' else 'out_nacks'] = out_nacks
        # Get InSatisfiedInterests of router at hand
        in_satisfied_interests = \
        rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InSatisfiedInterests')]['PacketRaw']
        in_satisfied_interests_list = in_satisfied_interests.to_list()
        in_satisfied_interests = sum(i for i in in_satisfied_interests)
        features[8 if mode == 'array' else 'in_interests'] = in_satisfied_interests
        # Get InTimedOutInterests of router at hand
        in_timedout_interests = \
        rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'InTimedOutInterests')]['PacketRaw']
        in_timedout_interests_list = in_timedout_interests.to_list()
        in_timedout_interests = sum(i for i in in_timedout_interests_list)
        features[9 if mode == 'array' else 'in_timedout_interests'] = in_timedout_interests
        # Get OutSatisfiedInterests of router at hand
        out_satisfied_interests = \
        rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutSatisfiedInterests')]['PacketRaw']
        out_satisfied_interests_list = out_satisfied_interests.to_list()
        out_satisfied_interests = sum(i for i in out_satisfied_interests_list)
        features[10 if mode == 'array' else 'out_satisfied_interests'] = out_satisfied_interests
        # Get OutTimedOutInterests of router at hand
        out_timedout_interests = \
        rate_data[(rate_data['Node'] == node_name) & (rate_data['Type'] == 'OutTimedOutInterests')]['PacketRaw']
        out_timedout_interests_list = out_timedout_interests.to_list()
        out_timedout_interests = sum(i for i in out_timedout_interests_list)
        features[11 if mode == 'array' else 'out_timedout_interests'] = out_timedout_interests
        # Return feature for node node_name
        # print('features: {}'.format(features))
        if mode == 'array':
            # print('np.count_nonzero(features): {}'.format(np.count_nonzero(features)))
            if np.isnan(features).any():
                raise ValueError('Something very wrong! All features are zeros!')
        elif mode == 'dict':
            nan_count = 0
            for key, value in enumerate(features):
                if math.isnan(value):
                    nan_count += 1
            if nan_count != 0:
                raise ValueError('Something very wrong! All features are zeros!')
        else:
            raise ValueError('Invalid mode for extracting node features!')

        return features

    def get_all_nodes_features(self, nodes_names, data):
        # Define empty list for nodes features
        nodes_features = {}
        # Iterate over each node and get their features
        for node_index, node_name in enumerate(nodes_names):
            features = self.get_node_features(data=data,
                                              node_name=node_name)
            nodes_features[node_name.split('Rout')[-1]] = features
        # print('nodes_features shape: {}'.format(nodes_features.shape))
        # Return nodes_features
        return nodes_features

    def insert_labels(self, graph, time, frequency):
        if self.scenario != 'normal':
            # CHeck if time of the current window is before or after the attack start time
            attack_is_on = True if time > self.time_att_start else False
        else:
            attack_is_on = False
        # If attack is on set graph label to 1 else to 0
        graph.graph['attack_is_on'] = attack_is_on
        # Append graph label corresponding to the simulation considered
        graph.graph['train_scenario'] = get_scenario_labels_dict()[self.scenario]
        # Set also time for debugging purposes
        graph.graph['time'] = time
        # Set also attack frequency for debugging purposes
        if self.scenario != 'normal':
            graph.graph['frequency'] = int(frequency)
        else:
            pass
        # Return graph with labels
        return graph

    def extract_graphs_from_simulation_files(self, simulation_files, simulation_index, total_simulations, split):
        # print('simulation_files: {}'.format(simulation_files))
        # Extract data from the considered simulation
        data = self.get_data(simulation_files)
        # Get names of nodes inside a simulation
        routers_names = self.get_router_names(data)
        # Define start time as one
        start_time = 1
        # Define empty list containing all graphs found in a simulation
        tg_graphs = []
        # Check if simulation has run up until the end or not. To avoid NaN issues inside features
        rate_trace_file = [file for file in simulation_files if 'rate-trace' in file][0]
        last_line_of_rate_trace_file = pd.read_csv(rate_trace_file, sep='\t', index_col=False).iloc[-1]
        simulation_time_from_rate_trace_file = last_line_of_rate_trace_file['Time']
        # Set simulation time depending on the last line of the trace file
        if simulation_time_from_rate_trace_file < self.simulation_time - 1:
            simulation_time = simulation_time_from_rate_trace_file - 1
        else:
            simulation_time = self.simulation_time - 1
        # For each index get the corresponding network traffic window and extract the features in that window
        for time in range(start_time, simulation_time + 1):
            if self.scenario != 'normal':
                # Print info
                frequency = simulation_files[0].split("/")[-2].split('x')[0]
                print(
                    "\r| Extracting {} split... |"
                    " Scenario: {} | Topology: {} |"
                    " Frequency: {} |"
                    " Simulation progress: {}/{} |"
                    " Time steps progress: {}/{} |".format(split,
                                                           self.scenario,
                                                           self.topology,
                                                           frequency,
                                                           simulation_index,
                                                           total_simulations,
                                                           time,
                                                           simulation_time),
                    end="\r")
            else:
                frequency = None
                print(
                    "\r| Extracting {} split... |"
                    " Scenario: {} | Topology: {} |"
                    " Simulation progress: {}/{} |"
                    " Time steps progress: {}/{} |".format(split,
                                                           self.scenario,
                                                           self.topology,
                                                           simulation_index,
                                                           total_simulations,
                                                           time,
                                                           simulation_time),
                    end="\r")
            if self.scenario == 'existing' and self.topology == 'dfn' and frequency == '32' \
                    and split == 'train' and simulation_index == 1 and time >= 299:
                continue
            # Get graph of the network during the current time window
            graph = self.get_graph_structure()
            filtered_data = self.filter_data_by_time(data, time)
            nodes_features = self.get_all_nodes_features(nodes_names=routers_names,
                                                         data=filtered_data)
            # Add nodes features to graph
            # print('graph.nodes: {}'.format(graph.nodes))
            # print('nodes_features: {}'.format(nodes_features))
            for node_name in graph.nodes:
                # print('node_name: {}'.format(node_name))
                # print('graph.nodes[node_name]: {}'.format(graph.nodes[node_name]))
                # print('nodes_features[node_name]: {}'.format(nodes_features[node_name]))
                graph.nodes[node_name]['x'] = nodes_features[node_name]
            # Debugging purposes
            # print('graph.nodes.data(): {}'.format(graph.nodes.data()))
            # Add labels to the graph as graph and nodes attributes
            graph = self.insert_labels(graph,
                                       time=time,
                                       frequency=frequency)
            # Debugging purposes
            # print('graph.graph: {}'.format(graph.graph))
            # print('graph.nodes.data(): {}'.format(graph.nodes.data()))
            # print('graph.edges.data(): {}'.format(graph.edges.data()))
            # Convert networkx graph into torch_geometric
            tg_graph = tg.utils.from_networkx(graph)
            # Add graph labels to the tg_graph
            for graph_label_name, graph_label_value in graph.graph.items():
                torch.tensor(graph_label_value, dtype=torch.int)
                tg_graph[graph_label_name] = torch.tensor(graph_label_value, dtype=torch.int)
            # print('tg_graph: {}'.format(tg_graph))
            # print('tg_graph.x: {}'.format(tg_graph.x))
            # print('tg_graph.edge_index: {}'.format(tg_graph.edge_index))
            # Append the graph for the current time window to the list of graphs
            tg_graphs.append(tg_graph)
        # Return the list of pytorch geometric graphs
        # print('tg_graphs: {}'.format(tg_graphs))
        return tg_graphs

    @staticmethod
    def store_tg_data_raw(raw_data, folder, file_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(raw_data,
                   os.path.join(folder, file_name))

    def split_files(self, files):
        # Split files depending on the train ids
        train_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.train_sim_ids]
        # print('train_files: {}'.format(train_files))
        val_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.val_sim_ids]
        # print('val_files: {}'.format(val_files))
        test_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.test_sim_ids]
        # print('test_files: {}'.format(test_files))
        return train_files, val_files, test_files

    @timeit
    def run(self, downloaded_data_file, raw_dir, raw_file_names):
        # print('\nraw_file_names: {}\n'.format(raw_file_names))
        # Split the received files into train, validation and test
        files_lists = self.split_files(downloaded_data_file)
        # Iterate over train validation and test and get graph samples
        print('Extracting graph data from each simulation of each split. This may take a while...')
        for index, files in enumerate(files_lists):
            list_of_tg_graphs = []
            # Iterate over frequencies
            frequencies = np.unique([file.split('/')[-2].split('x')[0] for file in files])
            # print('frequencies: {}'.format(frequencies))
            for frequence in frequencies:
                freq_files = [file for file in files if file.split('/')[-2].split('x')[0] == frequence]
                # print('freq_files: {}'.format(freq_files))
                # Iterating over index of simulations
                if index == 0:
                    split = 'train'
                    simulation_indices = self.train_sim_ids
                elif index == 1:
                    split = 'validation'
                    simulation_indices = self.val_sim_ids
                elif index == 2:
                    split = 'test'
                    simulation_indices = self.test_sim_ids
                else:
                    raise ValueError('Something went wrong with simulation indices')
                for s_index, simulation_index in enumerate(simulation_indices):
                    simulation_files = [file for file in freq_files if
                                        int(file.split('-')[-1].split('.')[0]) == simulation_index]
                    # Extract graphs from single simulation
                    tg_graphs = self.extract_graphs_from_simulation_files(simulation_files=simulation_files,
                                                                          simulation_index=s_index+1,
                                                                          total_simulations=len(simulation_indices),
                                                                          split=split)
                    # Add the graphs to the list of tg_graphs
                    list_of_tg_graphs += tg_graphs
            # Close info line
            print()
            # Store list of tg graphs in the raw folder of the tg dataset
            self.store_tg_data_raw(list_of_tg_graphs,
                                   raw_dir,
                                   raw_file_names[index])
