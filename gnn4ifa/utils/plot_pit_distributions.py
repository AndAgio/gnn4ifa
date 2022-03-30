# from gnn4ifa.data import IfaDataset
#
# dataset = IfaDataset(root='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data_tg',
#                      download_folder='/Users/andrea.agiollo/Documents/PhD/Projects/GNN-x-IFA/ifa_data')
# print('dataset.raw_dir: {}'.format(dataset.raw_dir))
# print('dataset.processed_dir: {}'.format(dataset.processed_dir))
import os
import pandas as pd
import numpy as np
import glob
import argparse
import itertools
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot_pit_distributions(download_folder, scenarios, topologies):
    # Define empty dictionary for pit sizes of topologies and scenarios
    pit_sizes = {topo: {scenario: {} for scenario in scenarios} for topo in topologies}
    print('pit_sizes: {}'.format(pit_sizes))
    # Iterate over each topology received in input
    for topology in topologies:
        # Iterate over each scenario passed as input
        for scenario in scenarios:
            assert scenario in ['normal', 'existing', 'non_existing']
            # Define the path to files containing data for current scenario
            path = simulations_path(download_folder=download_folder,
                                    scenario=scenario,
                                    topology=topology)
            print('path: {}'.format(path))
            # Get files list containing files of current scenario
            files_list = get_files_list(directory=path, scenario=scenario)
            # If the scenario is not the legitimate one then we need to plot one distribution for each frequency
            if scenario != 'normal':
                # Iterate over frequencies
                frequencies = np.unique([file.split('/')[-2].split('x')[0] for file in files_list])
                for frequence in frequencies:
                    freq_files = [file for file in files_list if file.split('/')[-2].split('x')[0] == frequence]
                    # Get pit distributions
                    pits = extract_pits_from_simulation_files(simulation_files=freq_files,
                                                              scenario=scenario,
                                                              simulation_time=300,
                                                              att_tim=50)
                    # Append distributions to dictionary for plotting
                    pit_sizes[topology][scenario][frequence] = pits
            else:
                # Get pit distributions
                pits = extract_pits_from_simulation_files(simulation_files=files_list,
                                                          scenario=scenario,
                                                          simulation_time=300,
                                                          att_tim=50)
                # Append distributions to dictionary for plotting
                pit_sizes[topology][scenario]['1'] = pits
    print('pit_sizes: {}'.format(pit_sizes))
    # Fit gaussian distributions
    gauss_dict = fit_gaussian(pit_sizes)
    print('gauss_dict: {}'.format(gauss_dict))
    # Plot distribution
    plot_gaussians(pit_sizes, gauss_dict)


def fit_gaussian(pit_sizes):
    gauss_dict = {
        topo: {scenario: {freq: {} for freq in pit_sizes[topo][scenario].keys()} for scenario in pit_sizes[topo].keys()}
        for topo in pit_sizes.keys()}
    # Iterate over each topology received in input
    for topo_name, topo_dict in pit_sizes.items():
        # Iterate over each scenario passed as input
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, data in scenario_dict.items():
                # Fit gaussian distribution over pit sizes
                mu, std = norm.fit(data)
                gauss_dict[topo_name][scenario_name][freq_name] = [mu, std, np.min(data), np.max(data)]
    return gauss_dict


def plot_gaussians(pit_sizes, gauss_dict):
    # Iterate over each topology received in input
    for topo_name, topo_dict in pit_sizes.items():
        # Define common plot for single topology
        combinations = list(itertools.combinations(list(topo_dict.keys()), 2))
        combinations = [comb for comb in combinations if 'normal' in comb]
        # print('combinations: {}'.format(combinations))
        fig, axs = plt.subplots(2, len(combinations), figsize=(15, 5))
        fig.suptitle('TOPOLOGY: {}'.format(topo_name.upper()))
        # Iterate over each combination and plot it
        for comb_index, comb in enumerate(combinations):
            # Iterate over each element of the combination
            for scenario in comb:
                # Iterate over frequencies
                for freq_name, data in topo_dict[scenario].items():
                    # Plot the histogram.
                    axs[0, comb_index].hist(data, bins=25, density=True, alpha=0.6,
                                            color=freq_color(freq_name),
                                            label='Freq = {}x'.format(freq_name))
                    title = r"Scenario: {}".format(comb[-1])
                    axs[0, comb_index].title.set_text(title)
                    # Plot the PDF.
                    # axs[0, comb_index].set_xlim(0, 2000)
                    axs[0, comb_index].set_ylim(0, 0.2)
                    axs[0, comb_index].legend()
                    x = np.linspace(0, 1200, 100000)
                    p = norm.pdf(x,
                                 gauss_dict[topo_name][scenario][freq_name][0],
                                 gauss_dict[topo_name][scenario][freq_name][1])
                    axs[1, comb_index].plot(x, p,
                                            color=freq_color(freq_name), linewidth=2,
                                            label='Freq = {}x'.format(freq_name))
                    axs[1, comb_index].set_ylim(0, 0.2)
                    axs[1, comb_index].legend()
        # Save generated graph image
        out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'pits_distributions')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        image_name = '{}.pdf'.format(topo_name)
        image_path = os.path.join(out_path, image_name)
        plt.savefig(image_path)
        plt.show()
        plt.close()


def freq_color(freq):
    freq_colors = {'1': 'green',
                   '4': 'black',
                   '8': 'tomato',
                   '16': 'orange',
                   '32': 'yellow',
                   '64': 'aqua',
                   '128': 'royalblue',
                   '256': 'violet'}
    return freq_colors[freq]


def simulations_path(download_folder, scenario, topology):
    return os.path.join(os.getcwd(), '..', download_folder,
                        'IFA_4_{}'.format(scenario) if scenario != 'normal' else scenario,
                        '{}_topology'.format(topology) if topology != 'dfn' else '{}_topology'.format(topology.upper()))


def get_files_list(directory, scenario):
    # Import stored dictionary of data
    if scenario != 'normal':
        file_names = glob.glob(os.path.join(directory, '*', '*.txt'))
    else:
        file_names = glob.glob(os.path.join(directory, '*.txt'))
    # print('file_names: {}'.format(file_names))
    return file_names


def extract_pits_from_simulation_files(simulation_files, scenario, simulation_time=300, att_tim=50):
    # print('simulation_files: {}'.format(simulation_files))
    # Extract data from the considered simulation
    data = get_data(simulation_files)
    # Get names of nodes inside a simulation
    routers_names = get_router_names(data)
    # Define start time as one
    start_time = 1
    # Define empty list containing all pits found in a simulation
    pits = []
    # Check if simulation has run up until the end or not. To avoid NaN issues inside features
    rate_trace_file = [file for file in simulation_files if 'rate-trace' in file][0]
    last_line_of_rate_trace_file = pd.read_csv(rate_trace_file, sep='\t', index_col=False).iloc[-1]
    simulation_time_from_rate_trace_file = last_line_of_rate_trace_file['Time']
    # Set simulation time depending on the last line of the trace file
    if simulation_time_from_rate_trace_file < simulation_time - 1:
        simulation_time = simulation_time_from_rate_trace_file - 1
    else:
        simulation_time = simulation_time - 1
    # For each index get the corresponding network traffic window and extract the features in that window
    for time in range(start_time, simulation_time + 1):
        # Filer data to get current time window
        filtered_data = filter_data_by_time(data, time)
        if scenario == 'normal' or time >= att_tim:
            pit_sizes = get_all_pit_sizes(nodes_names=routers_names,
                                          data=filtered_data)
            # Add pit sizes to pits
            pits += pit_sizes
    return pits


def get_router_names(data):
    # Get names of transmitter devices
    routers_names = data['rate']['Node'].unique()
    # Consider routers only
    routers_names = [i for i in routers_names if 'Rout' in i]
    # print('routers_names: {}'.format(routers_names))
    return routers_names


def filter_data_by_time(data, time, verbose=False):
    if verbose:
        print('Time: {}'.format(time))
    filtered_data = {}
    for key, value in data.items():
        filtered_data[key] = data[key][data[key]['Time'] == time]
    return filtered_data


def get_pit_size(data, node_name):
    # Get different modes of data
    pit_data = data['pit']
    # Get pit size of router at hand
    router_index = node_name.split('Rout')[-1]
    pit_size = pit_data[pit_data['Node'] == 'PIT_{}'.format(router_index)]['Size'].item()
    return pit_size


def get_all_pit_sizes(nodes_names, data, mode='list'):
    # Define empty list for nodes features
    if mode == 'list':
        pits = []
    else:
        pits = {}
    # Iterate over each node and get their features
    for node_index, node_name in enumerate(nodes_names):
        pit_size = get_pit_size(data=data,
                                node_name=node_name)
        if mode == 'list':
            pits += [pit_size]
        else:
            pits[node_name.split('Rout')[-1]] = pit_size
    # Return pit sizes
    return pits


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
            file = convert_pit_to_decent_format(file)
        # Read csv file
        file_data = pd.read_csv(file, sep='\t', index_col=False)
        # Put data in dictionary
        data[file_type] = file_data
    return data


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


def main():
    # Define scenarios for which the distribution plot is required
    download_folder = 'ifa_data'
    # Define scenarios for which the distribution plot is required
    scenarios = ['normal', 'existing', 'non_existing']
    # Define scenarios for which the distribution plot is required
    topologies = ['small', 'dfn']
    # Run distribution plotter
    plot_pit_distributions(download_folder, scenarios, topologies)


if __name__ == '__main__':
    main()
