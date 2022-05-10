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
import warnings

warnings.filterwarnings("ignore")


def plot_pit_distributions(download_folder, scenarios, topologies):
    # Define empty dictionary for pit sizes of topologies and scenarios
    # pit_sizes = {topo: {scenario: {} for scenario in scenarios} for topo in topologies}
    # # Get frequencies n_attackers combination
    # f_a_combinations = get_freq_att_combinations_from_file_names(download_folder)
    # for topology in topologies:
    #     for scenario in scenarios:
    #         if scenario != 'normal':
    #             for comb in f_a_combinations:
    #                 freq = comb[0]
    #                 n_att = comb[1]
    #                 if freq != '1' and n_att != '0':
    #                     pit_sizes[topology][scenario][freq] = {n_att: []}
    #         else:
    #             pit_sizes[topology][scenario]['1'] = {'0': []}
    pit_sizes = get_empty_dict_from_file_names(download_folder, scenarios)
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
            # print('path: {}'.format(path))
            # Get files list containing files of current scenario
            files_list = get_files_list(directory=path, scenario=scenario)
            # If the scenario is not the legitimate one then we need to plot one distribution for each frequency
            if scenario != 'normal':
                # Iterate over frequencies
                frequencies = np.unique([file.split('/')[-3].split('x')[0] for file in files_list])
                for frequence in frequencies:
                    freq_files = [file for file in files_list if file.split('/')[-3].split('x')[0] == frequence]
                    # Iterate over number of attackers
                    n_atts = set([file.split('/')[-2].split('_')[0] for file in freq_files])
                    for n_att in n_atts:
                        n_att_files = [file for file in freq_files if file.split('/')[-2].split('_')[0] == n_att]
                        # print('n_att_files: {}'.format(n_att_files))
                        # Get pit distributions
                        pits = extract_pits_from_simulation_files(simulation_files=n_att_files,
                                                                  scenario=scenario,
                                                                  simulation_time=300,
                                                                  att_tim=50)
                        # Append distributions to dictionary for plotting
                        pit_sizes[topology][scenario][frequence][n_att] += pits
            else:
                # Get pit distributions
                pits = extract_pits_from_simulation_files(simulation_files=files_list,
                                                          scenario=scenario,
                                                          simulation_time=300,
                                                          att_tim=50)
                # Append distributions to dictionary for plotting
                pit_sizes[topology][scenario]['1']['0'] += pits
    # print('pit_sizes: {}'.format(pit_sizes))
    # Fit gaussian distributions
    gauss_dict = fit_gaussian(pit_sizes)
    print('gauss_dict: {}'.format(gauss_dict))
    # Plot distribution
    plot_gaussians(pit_sizes, gauss_dict)
    plot_hists(pit_sizes, gauss_dict)
    plot_hists_and_gaussians(pit_sizes, gauss_dict)


def fit_gaussian(pit_sizes):
    gauss_dict = {topo: {scenario: {freq: {n_att: {}
                                           for n_att in pit_sizes[topo][scenario][freq].keys()}
                                    for freq in pit_sizes[topo][scenario].keys()}
                         for scenario in pit_sizes[topo].keys()}
                  for topo in pit_sizes.keys()}
    print('gauss_dict: {}'.format(gauss_dict))
    # Iterate over each topology received in input
    for topo_name, topo_dict in pit_sizes.items():
        # Iterate over each scenario passed as input
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over n attackers
                for n_att_name, data in freq_dict.items():
                    # Fit gaussian distribution over pit sizes
                    # print('topo: {}, scenario: {}, freq: {}, n_att: {}'.format(topo_name,
                    #                                                            scenario_name,
                    #                                                            freq_name,
                    #                                                            n_att_name))
                    # print('data: {}'.format(data))
                    mu, std = norm.fit(data)
                    gauss_dict[topo_name][scenario_name][freq_name][n_att_name] = [mu, std, np.min(data), np.max(data)]
    return gauss_dict


def plot_hists(pit_sizes, gauss_dict):
    # Iterate over each topology received in input
    n_topos = len(list(gauss_dict.keys()))
    fig, axs = plt.subplots(1, n_topos, figsize=(15, 5))
    axins = [axs[i].inset_axes([0.5, 0.5, 0.47, 0.47]) for i in range(n_topos)]
    fig.suptitle('PITs Distributions')
    # fig.suptitle('TOPOLOGY: {}'.format(topo_name.upper()))
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    label = 'F={}x, N={}'.format(freq_name, n_att_name) if scenario_name != 'normal' else 'Legitimate'
                    title = r"Topology: {}".format(topo_name.upper())
                    # Plot the histogram.
                    axs[topo_index].hist(data, bins=25, density=True, alpha=0.6,
                                         color=freq_color(freq_name),
                                         hatch=att_hatch(n_att_name),
                                         label=label)
                    axs[topo_index].title.set_text(title)
                    # Plot the PDF.
                    # axs[0, comb_index].set_xlim(0, 2000)
                    axs[topo_index].set_ylim(0, 0.2)
                    axs[topo_index].legend(ncol=3)
                    # inset axes....
                    axins[topo_index].hist(data, bins=25, density=True, alpha=0.6,
                                           color=freq_color(freq_name),
                                           hatch=att_hatch(n_att_name))
                    # sub region of the original image
                    x1, x2, y1, y2 = 0, 70, 0, 0.15
                    axins[topo_index].set_xlim(x1, x2)
                    axins[topo_index].set_ylim(y1, y2)
                    axins[topo_index].set_xticklabels([])
                    axins[topo_index].set_yticklabels([])
                    axs[topo_index].indicate_inset_zoom(axins[topo_index], edgecolor="black")

                    axs[topo_index].set_ylim(0, 0.2)
                    axs[topo_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    # Save generated graph image
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'pits_distributions')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image_name = 'PITS_hist_{}.pdf'.format([topo_name for topo_name, _ in gauss_dict.items()])
    image_path = os.path.join(out_path, image_name)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()
    plt.close()


def plot_gaussians(pit_sizes, gauss_dict):
    # Iterate over each topology received in input
    n_topos = len(list(gauss_dict.keys()))
    fig, axs = plt.subplots(1, n_topos, figsize=(15, 5))
    axins = [axs[i].inset_axes([0.5, 0.5, 0.47, 0.47]) for i in range(n_topos)]
    # fig.suptitle('PITs Distributions')
    # fig.suptitle('TOPOLOGY: {}'.format(topo_name.upper()))
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    label = 'F={}x, N={}'.format(freq_name, n_att_name) if scenario_name != 'normal' else 'Legitimate'
                    title = r"Topology: {}".format(topo_name.upper())
                    axs[topo_index].title.set_text(title)
                    x = np.linspace(0, 1200, 10000)
                    p = norm.pdf(x,
                                 gauss_dict[topo_name][scenario_name][freq_name][n_att_name][0],
                                 gauss_dict[topo_name][scenario_name][freq_name][n_att_name][1])
                    axs[topo_index].plot(x, p,
                                         color=freq_color(freq_name), linewidth=1,
                                         linestyle=att_line(n_att_name),
                                         # marker=att_marker(n_att_name),
                                         markersize=5,
                                         label=label)
                    axs[topo_index].fill_between(x, p, color=freq_color(freq_name), alpha=0.5)
                    # inset axes....
                    axins[topo_index].plot(x, p,
                                           color=freq_color(freq_name), linewidth=1,
                                           linestyle=att_line(n_att_name),
                                           # marker=att_marker(n_att_name),
                                           markersize=5)
                    axins[topo_index].fill_between(x, p, color=freq_color(freq_name), alpha=0.5)
                    # sub region of the original image
                    x1, x2, y1, y2 = 0, 70, 0, 0.15
                    axins[topo_index].set_xlim(x1, x2)
                    axins[topo_index].set_ylim(y1, y2)
                    axins[topo_index].set_xticklabels([])
                    axins[topo_index].set_yticklabels([])
                    axs[topo_index].indicate_inset_zoom(axins[topo_index], edgecolor="black")

                    axs[topo_index].set_ylim(0, 0.2)
                    axs[topo_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    # Save generated graph image
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'pits_distributions')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image_name = 'PITS_gauss_{}'.format([topo_name for topo_name, _ in gauss_dict.items()])
    image_path = os.path.join(out_path, image_name)
    plt.tight_layout()
    plt.savefig(image_path + '.pdf', dpi=200)
    plt.savefig(image_path + '.svg')
    plt.savefig(image_path + '.png')
    plt.savefig(image_path + '.jpg')
    plt.show()
    plt.close()
    # from pdfc.pdf_compressor import compress
    # compress(image_path + '.pdf', image_path + '_compressed.pdf', power=4)


def plot_hists_and_gaussians(pit_sizes, gauss_dict):
    # Iterate over each topology received in input
    n_topos = len(list(gauss_dict.keys()))
    fig, axs = plt.subplots(2, n_topos, figsize=(15, 5))
    axins = [[axs[i, j].inset_axes([0.5, 0.5, 0.47, 0.47]) for j in range(n_topos)] for i in range(2)]
    fig.suptitle('PITs Distributions')
    # fig.suptitle('TOPOLOGY: {}'.format(topo_name.upper()))
    topo_index = -1
    for topo_name, topo_dict in pit_sizes.items():
        topo_index += 1
        # Iterate over scenarios passed
        for scenario_name, scenario_dict in topo_dict.items():
            # Iterate over frequencies
            for freq_name, freq_dict in scenario_dict.items():
                # Iterate over number of attackers
                for n_att_name, data in freq_dict.items():
                    label = 'F={}x, N={}'.format(freq_name, n_att_name) if scenario_name != 'normal' else 'Legitimate'
                    title = r"Topology: {}".format(topo_name.upper())
                    axs[0, topo_index].title.set_text(title)
                    # Plot the histogram.
                    axs[0, topo_index].hist(data, bins=25, density=True, alpha=0.6,
                                            color=freq_color(freq_name),
                                            hatch=att_hatch(n_att_name),
                                            label=label)
                    axs[0, topo_index].title.set_text(title)
                    # inset axes....
                    axins[0][topo_index].hist(data, bins=25, density=True, alpha=0.6,
                                              color=freq_color(freq_name),
                                              hatch=att_hatch(n_att_name))
                    # sub region of the original image
                    x1, x2, y1, y2 = 0, 70, 0, 0.15
                    axins[0][topo_index].set_xlim(x1, x2)
                    axins[0][topo_index].set_ylim(y1, y2)
                    axins[0][topo_index].set_xticklabels([])
                    axins[0][topo_index].set_yticklabels([])
                    axs[0, topo_index].indicate_inset_zoom(axins[0][topo_index], edgecolor="black")
                    axs[0, topo_index].set_ylim(0, 0.2)
                    axs[0, topo_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
                    # Plot the PDF.
                    x = np.linspace(0, 1200, 100000)
                    p = norm.pdf(x,
                                 gauss_dict[topo_name][scenario_name][freq_name][n_att_name][0],
                                 gauss_dict[topo_name][scenario_name][freq_name][n_att_name][1])
                    axs[1, topo_index].plot(x, p,
                                            color=freq_color(freq_name), linewidth=1,
                                            linestyle=att_line(n_att_name),
                                            # marker=att_marker(n_att_name),
                                            markersize=5,
                                            label=label)
                    axs[1, topo_index].fill_between(x, p, color=freq_color(freq_name), alpha=0.5)
                    # inset axes....
                    axins[1][topo_index].plot(x, p,
                                              color=freq_color(freq_name), linewidth=1,
                                              linestyle=att_line(n_att_name),
                                              # marker=att_marker(n_att_name),
                                              markersize=5)
                    axins[1][topo_index].fill_between(x, p, color=freq_color(freq_name), alpha=0.5)
                    # sub region of the original image
                    x1, x2, y1, y2 = 0, 70, 0, 0.15
                    axins[1][topo_index].set_xlim(x1, x2)
                    axins[1][topo_index].set_ylim(y1, y2)
                    axins[1][topo_index].set_xticklabels([])
                    axins[1][topo_index].set_yticklabels([])
                    axs[1, topo_index].indicate_inset_zoom(axins[1][topo_index], edgecolor="black")

                    axs[1, topo_index].set_ylim(0, 0.2)
                    axs[1, topo_index].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    # Save generated graph image
    out_path = os.path.join(os.getcwd(), '..', 'output', 'plots', 'pits_distributions')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    image_name = 'PITS_hists_and_gauss_{}.pdf'.format([topo_name for topo_name, _ in gauss_dict.items()])
    image_path = os.path.join(out_path, image_name)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.show()
    plt.close()


def old_plot_gaussians(pit_sizes, gauss_dict):
    # Iterate over each topology received in input
    for topo_name, topo_dict in pit_sizes.items():
        # Define common plot for single topology
        combinations = list(itertools.combinations(list(topo_dict.keys()), 2))
        combinations = [comb for comb in combinations if 'normal' in comb]
        print('combinations: {}'.format(combinations))
        fig, axs = plt.subplots(2, len(combinations), figsize=(15, 10))
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


def att_line(att):
    att_lines = {'0': (0, ()),
                 '4': (0, (1, 1)),
                 '5': (0, (5, 5)),
                 '6': (0, (3, 5, 1, 5)),
                 '7': (0, (1, 10)),
                 '8': (0, (5, 10)),
                 '11': (0, (3, 10, 1, 10))}
    return att_lines[att]


def att_marker(att):
    att_markers = {'0': 'o',
                   '4': 'v',
                   '5': '^',
                   '6': '<',
                   '7': '>',
                   '8': '*',
                   '11': 's'}
    return att_markers[att]


def att_hatch(att):
    att_hatches = {'0': "/",
                   '4': "\\",
                   '5': "|",
                   '6': "-",
                   '7': "+",
                   '8': "x",
                   '11': "o", }
    return att_hatches[att]


def get_empty_dict_from_file_names(download_folder, scenarios):
    file_list = []
    for scenario in scenarios:
        if scenario == 'normal':
            file_list += glob.glob(
                os.path.join(os.getcwd(), '..', download_folder, scenario, '*', '*', '*', '*', '*.txt'))
        else:
            file_list += glob.glob(
                os.path.join(os.getcwd(), '..', download_folder, 'IFA_4_{}'.format(scenario), '*', '*', '*', '*',
                             '*.txt'))
    # print('file_list: {}'.format(file_list))
    empty_dict = {}
    combinations = []
    topologies = []
    for file in file_list:
        scenario = file.split('/')[-6].split('_')[-1].lower()
        topology = file.split('/')[-5].split('_')[0].lower()
        topologies.append(topology)
        freq = file.split('/')[-3].split('x')[0]
        n_att = file.split('/')[-2].split('_')[0]
        # print('topology: {}, scenario: {}, freq: {}, n_att: {}'.format(topology, scenario, freq, n_att))
        try:
            empty_dict[topology]
        except KeyError:
            empty_dict[topology] = {}
        try:
            empty_dict[topology][scenario]
        except KeyError:
            empty_dict[topology][scenario] = {}
        try:
            empty_dict[topology][scenario][freq]
        except KeyError:
            empty_dict[topology][scenario][freq] = {}
        try:
            empty_dict[topology][scenario][freq][n_att]
        except KeyError:
            empty_dict[topology][scenario][freq][n_att] = []
        # empty_dict.update({topology: {scenario: {freq: {n_att: []}}}})
    #     combinations.append((freq, n_att))
    # print(f'combinations: {combinations}')
    # combinations = list(set(combinations))
    # combinations.append(('1', '0'))
    # print(f'combinations: {combinations}')
    topologies = list(set(topologies))
    for topology in topologies:
        try:
            empty_dict[topology]['normal']
        except KeyError:
            empty_dict[topology]['normal'] = {}
        try:
            empty_dict[topology]['normal']['1']
        except KeyError:
            empty_dict[topology]['normal']['1'] = {}
        try:
            empty_dict[topology]['normal']['1']['0']
        except KeyError:
            empty_dict[topology]['normal']['1']['0'] = []
    # print('empty_dict: {}'.format(empty_dict))
    return empty_dict


def simulations_path(download_folder, scenario, topology):
    return os.path.join(os.getcwd(), '..', download_folder,
                        'IFA_4_{}'.format(scenario) if scenario != 'normal' else scenario,
                        '{}_topology'.format(topology) if topology != 'dfn' else '{}_topology'.format(topology.upper()))


def get_files_list(directory, scenario):
    # Import stored dictionary of data
    if scenario != 'normal':
        # print('ouaosdngoin: {}'.format(os.path.join(directory, '*', '*', '*', '*.txt')))
        file_names = glob.glob(os.path.join(directory, '*', '*', '*', '*.txt'))
    else:
        file_names = glob.glob(os.path.join(directory, '*.txt'))
    # print('file_names: {}'.format(file_names))
    return file_names


def get_simulation_time(simulation_files, simulation_time=300):
    # print('simulation_files: {}'.format(simulation_files))
    # Check if simulation has run up until the end or not. To avoid NaN issues inside features
    rate_trace_file = [file for file in simulation_files if 'rate-trace' in file][0]
    last_line_of_rate_trace_file = pd.read_csv(rate_trace_file, sep='\t', index_col=False).iloc[-1]
    simulation_time_from_rate_trace_file = last_line_of_rate_trace_file['Time']
    # Set simulation time depending on the last line of the trace file
    if simulation_time_from_rate_trace_file < simulation_time - 1:
        simulation_time = simulation_time_from_rate_trace_file - 1
    else:
        simulation_time = simulation_time - 1
    # Double check simulation time from the pit trace file
    try:
        pit_trace_file = [file for file in simulation_files if 'format-pit-size' in file][0]
    except IndexError:
        _ = get_data(simulation_files)
        simulation_files = glob.glob(os.path.join('/', os.path.join(*simulation_files[0].split('/')[:-1]), '*.txt'))
        pit_trace_file = [file for file in simulation_files if 'format-pit-size' in file][0]
    last_line_of_pit_trace_file = pd.read_csv(pit_trace_file, sep='\t', index_col=False).iloc[-1]
    simulation_time_from_pit_trace_file = last_line_of_pit_trace_file['Time'] - 1
    if simulation_time_from_pit_trace_file < simulation_time:
        simulation_time = simulation_time_from_pit_trace_file
    # Double check simulation time from the drop trace file
    drop_trace_file = [file for file in simulation_files if 'drop-trace' in file][0]
    last_line_of_drop_trace_file = pd.read_csv(drop_trace_file, sep='\t', index_col=False).iloc[-1]
    simulation_time_from_drop_trace_file = last_line_of_drop_trace_file['Time'] - 1
    # print(f'PIT last time: {simulation_time_from_pit_trace_file}')
    # print(f'Rate trace last time: {simulation_time_from_rate_trace_file}')
    # print(f'Drop trace last time: {simulation_time_from_drop_trace_file}')
    if simulation_time_from_drop_trace_file < simulation_time:
        simulation_time = simulation_time_from_drop_trace_file
    # print(f'simulation_time: {simulation_time}')
    return simulation_time


def extract_pits_from_simulation_files(simulation_files, scenario, simulation_time=300, att_tim=50):
    # print('simulation_files: {}'.format(simulation_files))
    simulation_files = rename_topology_files(simulation_files)
    # Extract data from the considered simulation
    data = get_data(simulation_files)
    # Get names of nodes inside a simulation
    routers_names = get_router_names(data)
    # Define start time as one
    start_time = 1
    # Define empty list containing all pits found in a simulation
    pits = []
    # Get simulation time. To avoid NaN issues inside features
    simulation_time = get_simulation_time(simulation_files, simulation_time=simulation_time)
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
    data = remove_topo_data_from_dict(data)
    # Get names of transmitter devices
    routers_names = data['rate']['Node'].unique()
    # Consider routers only
    routers_names = [i for i in routers_names if 'Rout' in i]
    # print('routers_names: {}'.format(routers_names))
    return routers_names


def filter_data_by_time(data, time, verbose=False):
    if verbose:
        print('Time: {}'.format(time))
    data = remove_topo_data_from_dict(data)
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


def remove_topo_data_from_dict(data):
    # print(f'data: {data}')
    data = {key: value for key, value in data.items() if key.split('/')[-1].split('-')[0] != 'topology'}
    # print(f'data: {data}')
    return data


def rename_topology_files(files):
    for index, file in enumerate(files):
        if '_topology' in file.split('/')[-1]:
            new_name = os.path.join('/',
                                    os.path.join(*file.split('/')[:-1]),
                                    '{}-{}.txt'.format(file.split('/')[-1].split('.')[0].split('_')[1],
                                                       file.split('/')[-1].split('.')[0].split('_')[0]))
            os.rename(file, new_name)
            files[index] = new_name
    return files


def get_data(files):
    # Define empty dictionary containing data
    data = {}
    # Iterate over all simulation files and read them
    for file in files:
        # print('file: {}'.format(file))
        # Check file type
        file_type = file.split('/')[-1].split('-')[0]
        if file_type == 'format':
            continue
        if file_type == 'pit':
            file = convert_pit_to_decent_format(file)
        if file_type == 'topology':
            # print('Converting topology file to decent format...')
            file = convert_topology_to_decent_format(file)
        # Read csv file
        file_data = pd.read_csv(file, sep='\t', index_col=False)
        # Put data in dictionary
        data[file_type] = file_data
    return data


def get_lines_from_unformatted_topology_file(file):
    reversed_lines = reversed(list(open(file)))
    keep = []
    # Cycle through file from bottom to top
    for line in reversed_lines:
        if line.rstrip() == '':
            continue
        if line.rstrip()[0] == '#':
            break
        else:
            keep.append('\t'.join(line.split()) + '\n')
    keep = reversed(keep)
    return keep


def convert_topology_to_decent_format(file):
    final_lines = get_lines_from_unformatted_topology_file(file)
    # Store file with new name
    new_file = os.path.join('/', os.path.join(*file.split('/')[:-1]),
                            'format-{}'.format(file.split('/')[-1]))
    if os.path.exists(new_file):
        os.remove(new_file)
    with open(new_file, 'w') as f:
        f.write('Source\tDestination\n')
        for item in final_lines:
            f.write(item)
    return new_file


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
    scenarios = ['normal', 'existing']  # , 'non_existing']
    # Define scenarios for which the distribution plot is required
    topologies = ['small', 'dfn']
    # Run distribution plotter
    plot_pit_distributions(download_folder, scenarios, topologies)


if __name__ == '__main__':
    main()
