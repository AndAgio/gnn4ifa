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
from gnn4ifa.utils import timeit

warnings.filterwarnings("ignore")


class Extractor():
    def __init__(self, data_dir='ifa_data',
                 tg_dataset_path='ifa_data_tg',
                 scenario='existing',
                 topology='small',
                 train_sim_ids=[1, 2, 3],
                 val_sim_ids=[4],
                 test_sim_ids=[5],
                 simulation_time=1500,
                 time_att_start=10):
        self.data_dir = data_dir
        self.tg_dataset_path = tg_dataset_path

        self.scenario = scenario
        self.topology = topology

        self.train_sim_ids = train_sim_ids
        self.val_sim_ids = val_sim_ids
        self.test_sim_ids = test_sim_ids

        self.simulation_time = simulation_time
        self.time_att_start = time_att_start

    def read_files(self):
        # Getting files depending on data directory, scenario and simulation time chosen
        filenames = glob.glob(os.path.join(os.getcwd(), self.data_dir, '16_Nodes_Dataset', '*',
                                           'Packet_Trace_' + str(int(self.simulation_time)) + 's', '*.csv'))
        filenames.sort()
        # print('filenames: {}'.format(filenames))
        return filenames

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
            print('file_data: {}'.format(file_data))
            # Put data in dictionary
            data[file_type] = file_data
        print('data: {}'.format(data))
        return data

    @staticmethod
    def convert_pit_to_decent_format(file):
        print('\n\nfile: {}\n\n'.format(file))
        # Get number of lines of file
        num_lines = sum(1 for _ in open(file))
        # Get index of lines containing time
        times_lines = [i for i, line in enumerate(open(file)) if 'Simulation time' in line]
        print('times_lines: {}'.format(times_lines))
        # Get number of routers
        n_routers = times_lines[1] - 1
        # Get lines containing actual data
        data_lines = [line for line in open(file) if 'Simulation time' not in line]
        print('data_lines: {}'.format(data_lines))
        # Append timing to each line of the data_lines
        final_lines = []
        for i, line in enumerate(open(file)):
            if 'Simulation time' not in line:
                time = i // (n_routers + 1) + 1
                final_lines.append(str(time) + '\t' + line.replace(' ', '\t'))
        print('final_lines: {}'.format(final_lines))
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
        return routers_names

    def apply_time_window(self, data, index, full_data=False):
        if (full_data):
            start_time = self.time_start
        else:
            start_time = index * self.time_window + self.time_start
        end_time = (index + 1) * self.time_window + self.time_start
        # Get all packets that have been sent at the physical layer between start and end time (depending on window size)
        condition = (data[self.send_time_feat_micro] > start_time) & (data[self.send_time_feat_micro] <= end_time)
        sent_packets_sequence = data[condition]
        # Get all packets that have been received at the physical layer between start and end time (depending on window size)
        condition = (data[self.rec_time_feat_micro] > start_time) & (data[self.rec_time_feat_micro] <= end_time)
        rec_packets_sequence = data[condition]
        return sent_packets_sequence, rec_packets_sequence

    def extract_data_up_to(self, data, time, verbose=False):
        if verbose:
            print('Time: {}'.format(time))
        # From the pandas dataframe extract only those packets arrived up to a certain second
        condition = (data[self.rec_time_feat_micro] <= time)
        data = data[condition]
        return data

    @staticmethod
    def refine_edges_list(edges):
        edges_to_remove = list()
        # For each edges in the list of edges check if two edges have the same source
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                source_node_first_link = edges[i][0]
                source_node_second_link = edges[j][0]
                # If they have the same source then the oldest link must be removed since it means that the node had changed parent
                if (source_node_first_link == source_node_second_link):
                    edges_to_remove.append(edges[i])
        # Get a set from the list of edges to remove and remove all of them from the original edges list
        edges_to_remove = set(edges_to_remove)
        for ed in edges_to_remove:
            edges.remove(ed)
        # Return refined edges
        return edges

    def get_dodag(self, data, debug=False):
        # Get source ids and dest ids only with numbers
        source_ids = data['SOURCE_ID'].values.tolist()
        # source_ids = [id.split("-")[-1] for id in source_ids]
        dest_ids = data['DESTINATION_ID'].values.tolist()
        # dest_ids = [id.split("-")[-1] for id in dest_ids]
        # Each DAO represents a potential edge
        edges = [(source_ids[i], dest_ids[i]) for i in range(len(source_ids))]
        edges = [edge for edge in edges if str(edge[0]) != 'nan' and ('SENSOR' in edge[0] or 'SINKNODE' in edge[0])
                 and str(edge[1]) != 'nan' and ('SENSOR' in edge[1] or 'SINKNODE' in edge[1])]
        # Remove duplicate potential edges and maintain the order
        seen = set()
        seen_add = seen.add
        edges = [x for x in edges if not (x in seen or seen_add(x))]
        # Get list of nodes names (order doesn't matter)
        source_ids = self.get_unique_nodes_names(data, key='SOURCE_ID')  # list(dict.fromkeys(source_ids))
        # source_ids = [id.split("-")[-1] for id in source_ids]
        dest_ids = self.get_unique_nodes_names(data, key='DESTINATION_ID')  # list(dict.fromkeys(dest_ids))
        # dest_ids = [id.split("-")[-1] for id in dest_ids]
        list_of_nodes = source_ids + dest_ids
        list_of_nodes = list(dict.fromkeys(list_of_nodes))
        # Use nx to obtain the graph corresponding to the dodag
        dodag = nx.DiGraph()
        # Refine the list of edges in order to keep only most recent father-son relationships
        edges = self.refine_edges_list(edges)
        # Build the DODAG graph from nodes and edges lists
        dodag.add_nodes_from(list_of_nodes)
        dodag.add_edges_from(edges)
        # Print and plot for debugging purposes
        if debug:
            print('dodag: {}'.format(dodag))
            print('dodag.nodes: {}'.format(dodag.nodes))
            print('dodag.edges: {}'.format(dodag.edges))
            subax1 = plt.subplot(111)
            nx.draw(dodag, with_labels=True, font_weight='bold')
            plt.show()
        # Return networkx dodag
        return dodag

    @staticmethod
    def get_node_features(rec_packets_data, sent_packets_data, node_name, mode='array'):
        if mode == 'array':
            features = np.zeros((15), dtype=float)
        elif mode == 'dict':
            features = {}
        else:
            raise ValueError('Invalid mode for extracting node features!')
        # Get received and sent packets
        received_packets = rec_packets_data[rec_packets_data['RECEIVER_ID'] == node_name]
        transmitted_packets = sent_packets_data[sent_packets_data['TRANSMITTER_ID'] == node_name]
        # Get number of DIO received
        received_count = received_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts()
        if ('DIO' in received_count):
            features[0 if mode == 'array' else 'rec_DIO'] = received_count['DIO']
        else:
            features[0 if mode == 'array' else 'rec_DIO'] = 0
        # Get number of DIO transmitted
        transmitted_count = transmitted_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts()
        if ('DIO' in transmitted_count):
            features[1 if mode == 'array' else 'tx_DIO'] = transmitted_count['DIO']
        else:
            features[1 if mode == 'array' else 'tx_DIO'] = 0
        # Get number of DAO received
        if ('DAO' in received_count):
            features[2 if mode == 'array' else 'rec_DAO'] = received_count['DAO']
        else:
            features[2 if mode == 'array' else 'rec_DAO'] = 0
        # Get number of DAO transmitted
        if ('DAO' in transmitted_count):
            features[3 if mode == 'array' else 'tx_DAO'] = transmitted_count['DAO']
        else:
            features[3 if mode == 'array' else 'tx_DAO'] = 0
        # Get number of DIS received
        if ('DIS' in received_count):
            features[4 if mode == 'array' else 'rec_DIS'] = received_count['DIS']
        else:
            features[4 if mode == 'array' else 'rec_DIS'] = 0
        # Get number of DAO transmitted
        if ('DIS' in transmitted_count):
            features[5 if mode == 'array' else 'tx_DIS'] = transmitted_count['DIS']
        else:
            features[5 if mode == 'array' else 'tx_DIS'] = 0
        # Get number of application packets received
        received_app_count = received_packets['PACKET_TYPE'].value_counts()
        if ('Control_Packet' in received_app_count):
            features[6 if mode == 'array' else 'rec_APP'] = received_app_count.sum() - received_app_count[
                'Control_Packet']
        else:
            features[6 if mode == 'array' else 'rec_APP'] = received_app_count.sum()
        # Get number of application packets transmitted
        transmitted_app_count = transmitted_packets['PACKET_TYPE'].value_counts()
        if ('Control_Packet' in transmitted_app_count):
            features[7 if mode == 'array' else 'tx_APP'] = transmitted_app_count.sum() - transmitted_app_count[
                'Control_Packet']
        else:
            features[7 if mode == 'array' else 'tx_APP'] = transmitted_app_count.sum()
        # Get all packets received/transmitted by the node
        all_packets = pd.concat([rec_packets_data, sent_packets_data], axis=0)
        app_list = all_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts().index.to_list()
        control_list = ['DIO', 'DAO', 'DIS', 'DAO-ACK', 'OSPF_HELLO',
                        'OSPF_DD', 'OSPF_LSREQ', 'OSPF_LSUPDATE', 'OSPF_LSACK']
        for control in control_list:
            if (control in app_list):
                app_list.remove(control)
        features[8 if mode == 'array' else 'n_all'] = len(app_list)
        # Get number of different source IPs
        source_ips = all_packets['SOURCE_IP'].value_counts().index.to_list()
        features[9 if mode == 'array' else 'n_source_IPs'] = len(source_ips)
        # Get number of different destination IPs
        destination_ips = all_packets['DESTINATION_IP'].value_counts().index.to_list()
        features[10 if mode == 'array' else 'n_destination_IPs'] = len(destination_ips)
        # Get number of Gateway IPs
        gateway_ips = transmitted_packets['GATEWAY_IP'].value_counts().index.to_list()
        features[11 if mode == 'array' else 'n_gateway_IPs'] = len(gateway_ips)
        # Get successful transmission rate
        successful = transmitted_packets[transmitted_packets['PACKET_STATUS'] == 'Successful']
        collided = transmitted_packets[transmitted_packets['PACKET_STATUS'] == 'Collided']
        if (len(transmitted_packets.index) == 0):
            rate = 1
        else:
            rate = len(successful.index) / len(transmitted_packets.index)
        features[12 if mode == 'array' else 'succ_rate'] = rate
        # Get number of broadcast packets sent
        broadcast = transmitted_packets[transmitted_packets['DESTINATION_IP'] == 'FF00:0:0:0:0:0:0:0']
        features[13 if mode == 'array' else 'n_broad'] = len(broadcast.index)
        # Get number of incoming application packets that do not have itself as destination
        received_app_pcks = received_packets[received_packets['PACKET_TYPE'] == 'Sensing']
        incoming = received_app_pcks[(received_app_pcks['DESTINATION_ID'] != received_app_pcks['RECEIVER_ID']) & (
                received_app_pcks['PACKET_STATUS'] == 'Successful')]
        n_incoming_app_pcks = len(incoming.index)
        # Get number of outgoing application packets that do not have itself as source
        transmitted_app_pcks = transmitted_packets[transmitted_packets['PACKET_TYPE'] == 'Sensing']
        outgoing = transmitted_app_pcks[
            transmitted_app_pcks['SOURCE_ID'] != transmitted_app_pcks['TRANSMITTER_ID']]
        n_outgoing_app_pcks = len(outgoing.index)
        if (n_incoming_app_pcks == n_outgoing_app_pcks):
            features[14 if mode == 'array' else 'inc_vs_out'] = 1
        elif (n_outgoing_app_pcks != 0 and n_incoming_app_pcks == 0):
            features[14 if mode == 'array' else 'inc_vs_out'] = 1
        else:
            features[14 if mode == 'array' else 'inc_vs_out'] = n_outgoing_app_pcks / n_incoming_app_pcks  # * 100
        # Return feature for node node_name
        return features

    def get_all_nodes_features(self, nodes_names, sent_packets_data, rec_packets_data):
        # Define empty list for nodes features
        nodes_features = {}
        # Iterate over each node and get their features
        for node_index, node_name in enumerate(nodes_names):
            # print('node_name: {}'.format(node_name))
            features = self.get_node_features(rec_packets_data=rec_packets_data,
                                              sent_packets_data=sent_packets_data,
                                              node_name=node_name)
            nodes_features[node_name] = features
        # print('nodes_features: {}'.format(nodes_features))
        # Convert nodes features into numpy array and tensorize it
        # nodes_features = np.asarray(nodes_features)
        # print('nodes_features: {}'.format(nodes_features))
        # nodes_features = torch.tensor(nodes_features)
        # print('nodes_features: {}'.format(nodes_features))
        # print('nodes_features shape: {}'.format(nodes_features.shape))
        # Return nodes_features
        return nodes_features

    @staticmethod
    def get_edge_features(packets_data, edge, mode='array'):
        # Define set of empty features depending on the extraction mode
        if mode == 'array':
            features = np.zeros((6), dtype=float)
        elif mode == 'dict':
            features = {}
        else:
            raise ValueError('Invalid mode for extracting node features!')
        # Unpack edge to get sender and receiver IDs
        transmitter_id, receiver_id = edge
        # Get packets belonging only to the considered edge
        condition = (packets_data['TRANSMITTER_ID'] == transmitter_id) & (packets_data['RECEIVER_ID'] == receiver_id)
        edge_packets = packets_data[condition]
        # print('edge_packets: {}'.format(edge_packets))
        # Get number of DIOs
        control_count = edge_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts()
        if ('DIO' in control_count):
            features[0 if mode == 'array' else 'DIOs'] = control_count['DIO']
        else:
            features[0 if mode == 'array' else 'DIOs'] = 0
        # Get number of DAOs
        if ('DAO' in control_count):
            features[1 if mode == 'array' else 'DAOs'] = control_count['DAO']
        else:
            features[1 if mode == 'array' else 'DAOs'] = 0
        # Get number of DISs
        if ('DIS' in control_count):
            features[2 if mode == 'array' else 'DISs'] = control_count['DIS']
        else:
            features[2 if mode == 'array' else 'DISs'] = 0
        # Get number of application packets
        app_count = edge_packets['PACKET_TYPE'].value_counts()
        if ('Control_Packet' in app_count):
            features[3 if mode == 'array' else 'APPs'] = app_count.sum() - app_count['Control_Packet']
        else:
            features[3 if mode == 'array' else 'APPs'] = app_count.sum()
        # Get all packets received/transmitted by the node
        app_list = edge_packets['CONTROL_PACKET_TYPE/APP_NAME'].value_counts().index.to_list()
        control_list = ['DIO', 'DAO', 'DIS', 'DAO-ACK', 'OSPF_HELLO',
                        'OSPF_DD', 'OSPF_LSREQ', 'OSPF_LSUPDATE', 'OSPF_LSACK']
        for control in control_list:
            if (control in app_list):
                app_list.remove(control)
        features[4 if mode == 'array' else 'n_all'] = len(app_list)
        # Get successful transmission rate
        successful = edge_packets[edge_packets['PACKET_STATUS'] == 'Successful']
        collided = edge_packets[edge_packets['PACKET_STATUS'] == 'Collided']
        if (len(edge_packets.index) == 0):
            rate = 100
        else:
            rate = len(successful.index) / len(edge_packets.index) * 100
        features[5 if mode == 'array' else 'succ_rate'] = rate
        # Return feature for node node_name
        return features

    def get_all_edges_features(self, edges, packets_data):
        # Define empty list for edges features
        edges_features = {}
        # Iterate over each edge of the dodag and get their features
        for _, edge in enumerate(edges):
            # print('edge: {}'.format(edge))
            features = self.get_edge_features(packets_data=packets_data,
                                              edge=edge)
            edges_features[edge] = features
        # Return nodes_features
        return edges_features

    def insert_labels(self, dodag, filename, time):
        # Define labels dictionary for attack -> index
        attacks_labels_dict = get_scenario_labels_dict()
        # Extract scenario name and simulation name
        scenario = filename.split('/')[-3]
        simulation_name = filename.split('/')[-1].split('.')[0]
        # Read file containing labels
        labels_file_name = os.path.join(os.getcwd(),
                                        self.data_dir,
                                        '16_Nodes_Dataset',
                                        scenario,
                                        'Packet_Trace_{}s'.format(int(self.simulation_time)),
                                        'attacks_start_time.txt')
        # print('labels_file_name: {}'.format(labels_file_name))
        if scenario == 'Wormhole':
            names = ['sim_name', 'scenario', 'not1', 'not2', 'not3', 'not4', 'att1', 'not5', 'att2', 'not6', 'not7',
                     'not8', 'not9', 'start_time']
        elif scenario in ['Local_Repair', 'Worst_Parent']:
            names = ['sim_name', 'scenario1', 'scenario2', 'scenario3', 'not1', 'not2', 'not3', 'not4', 'att1', 'not6',
                     'not7', 'not8', 'not9',
                     'start_time']
        elif '_' in scenario or scenario in ['DIS', 'Rank', 'Version']:
            names = ['sim_name', 'scenario1', 'scenario2', 'not1', 'not2', 'not3', 'not4', 'att1', 'not6', 'not7',
                     'not8', 'not9',
                     'start_time']
        else:
            names = ['sim_name', 'scenario', 'not1', 'not2', 'not3', 'not4', 'att1', 'not6', 'not7', 'not8', 'not9',
                     'start_time']
        if scenario != 'Legitimate':
            labels_file = pd.read_csv(labels_file_name,
                                      sep=" ",
                                      header=None,
                                      names=names)
            # print('labels_file: {}'.format(labels_file))
            # Extract relevant labels for the current simulation
            labels_line = labels_file[labels_file['sim_name'] == '{}:'.format(simulation_name)].reset_index(drop=True)
            # print('Labels: {}'.format(labels_line))
            attack_start_time = labels_line.at[0, 'start_time']
            # print('attack_start_time: {}'.format(attack_start_time))
            if scenario != 'Wormhole':
                attacker_ids = [labels_line.at[0, 'att1']]
            else:
                attacker_ids = [labels_line.at[0, 'att1'], labels_line.at[0, 'att2']]
            # print('attacker_ids: {}'.format(attacker_ids))
            # CHeck if time of the current window is before or after the attack start time
            attack_is_on = True if time > attack_start_time / 1e6 else False
        else:
            attack_is_on = False
        # If attack is on set graph label to 1 else to 0
        dodag.graph['attack_is_on'] = attack_is_on
        # Append graph label corresponding to the simulation considered
        dodag.graph['scenario'] = attacks_labels_dict[scenario]
        # Define nodes labels
        attackers = {node_name: {'attacker': 0} for node_name in dodag.nodes}
        if simulation_name != 'Legitimate' and attack_is_on:
            for attacker_id in attacker_ids:
                node_name = 'SENSOR-{}'.format(
                    attacker_id) if attacker_id < dodag.number_of_nodes() else 'SINKNODE-{}'.format(attacker_id)
                attackers[node_name]['attacker'] = 1.
        # print('attackers: {}'.format(attackers))
        # Append attackers labels to dodag as nodes features
        nx.set_node_attributes(dodag, attackers)
        # Return dodag with labels
        return dodag

    def extract_graphs_from_simulation_files(self, simulation_files, simulation_index, total_simulations):
        print('simulation_files: {}'.format(simulation_files))
        # Extract data from the considered simulation
        data = self.get_data(simulation_files)
        # Get names of nodes inside a simulation
        routers_names = self.get_router_names(data)
        print('routers_names: {}'.format(routers_names))
        # Define start time as one
        start_time = 1
        # Define empty list containing all graphs found in a simulation
        tg_graphs = []
        # For each index get the corresponding network traffic window and extract the features in that window
        for time in range(start_time, self.simulation_time + 1):
            # Print info
            frequency = simulation_files[0].split("/")[-2].split('x')[0]
            print(
                "\r| Scenario: {} | Topology: {} | Frequency: {} |\t"
                "Simulation progress: {}/{} |\t"
                "Time steps progress: {}/{} |".format(self.scenario,
                                                      self.topology,
                                                      frequency,
                                                      simulation_index + 1,
                                                      total_simulations,
                                                      time,
                                                      self.simulation_time),
                end="\r")
            # Get dodag of the network during the current time window
            dodag = self.get_dodag(self.extract_data_up_to(data, (index + 1) * self.time_window))
            sent_packets_sequence, rec_packets_sequence = self.apply_time_window(data, index, full_data=False)
            nodes_features = self.get_all_nodes_features(nodes_names=routers_names,
                                                         sent_packets_data=sent_packets_sequence,
                                                         rec_packets_data=rec_packets_sequence)
            # Add nodes features to dodag
            for node_name in dodag.nodes:
                # print('node_name: {}'.format(node_name))
                # print('dodag.nodes[node_name]: {}'.format(dodag.nodes[node_name]))
                # print('nodes_features[node_name]: {}'.format(nodes_features[node_name]))
                dodag.nodes[node_name]['x'] = nodes_features[node_name]
            # Debugging purposes
            # print('dodag.nodes.data(): {}'.format(dodag.nodes.data()))
            # Extract features for edges of the graph
            edges_features = self.get_all_edges_features(edges=dodag.edges,
                                                         packets_data=sent_packets_sequence)
            # Add edge features to dodag
            for edge in dodag.edges:
                dodag.edges[edge]['edge_attr'] = edges_features[edge]
            # Debugging purposes
            # print('dodag.edges.data(): {}'.format(dodag.edges.data()))
            # Add labels to the dodag as graph and nodes attributes
            dodag = self.insert_labels(dodag,
                                       filename=filenames[file_index],
                                       time=time)
            # Debugging purposes
            # print('dodag.graph: {}'.format(dodag.graph))
            # print('dodag.nodes.data(): {}'.format(dodag.nodes.data()))
            # print('dodag.edges.data(): {}'.format(dodag.edges.data()))
            # Convert networkx graph into torch_geometric
            tg_graph = tg.utils.from_networkx(dodag)
            # Add graph labels to the tg_graph
            for graph_label_name, graph_label_value in dodag.graph.items():
                torch.tensor(graph_label_value, dtype=torch.int)
                tg_graph[graph_label_name] = torch.tensor(graph_label_value, dtype=torch.int)
            # print('tg_graph: {}'.format(tg_graph))
            # Append the graph for the current time window to the list of graphs
            tg_graphs.append(tg_graph)
        # Return the list of pytorch geometric graphs
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
        print('train_files: {}'.format(train_files))
        val_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.val_sim_ids]
        print('val_files: {}'.format(val_files))
        test_files = [file for file in files if int(file.split('-')[-1].split('.')[0]) in self.test_sim_ids]
        print('test_files: {}'.format(test_files))
        return train_files, val_files, test_files

    @timeit
    def run(self, downloaded_data_file, raw_dir, raw_file_names):
        print('\nraw_file_names: {}\n'.format(raw_file_names))
        # Split the received files into train, validation and test
        files_lists = self.split_files(downloaded_data_file)
        # Iterate over train validation and test and get graph samples
        print('Extracting graph data from each simulation of each split. This may take a while...')
        list_of_tg_graphs = []
        for index, files in enumerate(files_lists):
            # Iterate over frequencies
            frequencies = np.unique([file.split('/')[-2].split('x')[0] for file in files])
            print('frequencies: {}'.format(frequencies))
            for frequence in frequencies:
                freq_files = [file for file in files if file.split('/')[-2].split('x')[0] == frequence]
                print('freq_files: {}'.format(freq_files))
                # Iterating over index of simulations
                if index == 0:
                    print('Extracting train split...')
                    simulation_indices = self.train_sim_ids
                elif index == 1:
                    print('Extracting validation split...')
                    simulation_indices = self.val_sim_ids
                elif index == 2:
                    print('Extracting test split...')
                    simulation_indices = self.test_sim_ids
                else:
                    raise ValueError('Something went wrong with simulation indices')
                for simulation_index in simulation_indices:
                    simulation_files = [file for file in freq_files if
                                        int(file.split('-')[-1].split('.')[0]) == simulation_index]
                    # Extract graphs from single simulation
                    tg_graphs = self.extract_graphs_from_simulation_files(simulation_files=simulation_files,
                                                                          simulation_index=simulation_index,
                                                                          total_simulations=len(simulation_indices))
                    # Add the graphs to the list of tg_graphs
                    list_of_tg_graphs += tg_graphs
            # Close info line
            print()
            # Store list of tg graphs in the raw folder of the tg dataset
            self.store_tg_data_raw(list_of_tg_graphs,
                                   raw_dir,
                                   raw_file_names[index])
