import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_weights(file_path, mode):
    result = {}

    if mode == 1:
        # Load LSTM
        # Format {r1,r2,r3}
        dict_by_nodes = \
            {'deep': {'m1': {'large': {}, 'medium': {}, 'small': {}},
                      'm2': {'large': {}, 'medium': {}, 'small': {}},
                      'm3': {'large': {}, 'medium': {}, 'small': {}}},
             'shallow': {'m1': {'large': {}, 'medium': {}, 'small': {}},
                         'm2': {'large': {}, 'medium': {}, 'small': {}},
                         'm3': {'large': {}, 'medium': {}, 'small': {}}}}
        dict_by_nodes_f1_only = \
            {'deep': {'m1': {'large': {}, 'medium': {}, 'small': {}},
                      'm2': {'large': {}, 'medium': {}, 'small': {}},
                      'm3': {'large': {}, 'medium': {}, 'small': {}}},
             'shallow': {'m1': {'large': {}, 'medium': {}, 'small': {}},
                         'm2': {'large': {}, 'medium': {}, 'small': {}},
                         'm3': {'large': {}, 'medium': {}, 'small': {}}}}
        # dict_by_mode = {'m1': {}, 'm2': {}, 'm3': {}}
        # dict_by_scale = {}
        # dict_by_ratio = {}
        folders = ['pre_trained_128', 'pre_trained_256']

        for filename in folders:
            if filename.split('_')[-1] == '128':
                depth_str = 'shallow'
            elif filename.split('_')[-1] == '256':
                depth_str = 'deep'
            else:
                continue
            modes = os.listdir(file_path + filename)

            for mode in modes:
                if mode == '.DS_Store':
                    continue
                mode_str = mode[4]

                folder_1 = file_path + filename + '/' + mode
                scales = os.listdir(folder_1)

                for scale in scales:
                    if scale == '.DS_Store':
                        continue
                    scale_str = scale

                    folder_2 = folder_1 + '/' + scale
                    pickles = os.listdir(folder_2)

                    step_dict = {}
                    step_dict_f1 = {}

                    for pk in pickles:
                        if mode_str == '1' and depth_str == 'deep':
                            print('here')

                        if 'optimal' in pk:
                            folder_3 = folder_2 + '/' + pk
                            md = pk[14]
                            if 'medium' in pk:
                                ratio = int(pk[36])
                            else:
                                ratio = int(pk[35])
                            if md == mode_str:
                                with open(folder_3, 'rb') as f:
                                    params = pickle.load(f)
                                    step_dict[ratio] = params
                                    step_dict_f1[ratio] = params['f1']
                            else:
                                print('error occurred')
                    dict_by_nodes[depth_str]['m' + mode_str][scale_str] = step_dict
                    dict_by_nodes_f1_only[depth_str]['m' + mode_str][scale_str] = step_dict_f1

            print('x')

        return dict_by_nodes, dict_by_nodes_f1_only
    elif mode == 2:
        # Load LR
        pass
    else:
        return result


def draw_ratio_plot(ratio):
    # Y-axis: F1
    # X-axis: Scale
    # Legend: Ratios
    # Y vs ratios
    y_1 = np.zeros([3])
    y_2 = np.zeros([3])
    y_3 = np.zeros([3])
    x = np.arange(1, 4, 1)
    for key in ratio:
        if key == 'large':
            for ratio in ratio[key]:
                if ratio == 1:
                    y_1[2] = ratio[key][ratio]
                elif ratio == 2:
                    y_2[2] = ratio[key][ratio]
                elif ratio == 3:
                    y_3[2] = ratio[key][ratio]
        elif key == 'medium':
            for ratio in ratio[key]:
                if ratio == 1:
                    y_1[1] = ratio[key][ratio]
                elif ratio == 2:
                    y_2[1] = ratio[key][ratio]
                elif ratio == 3:
                    y_3[1] = ratio[key][ratio]
        else:
            for ratio in ratio[key]:
                if ratio == 1:
                    y_1[0] = ratio[key][ratio]
                elif ratio == 2:
                    y_2[0] = ratio[key][ratio]
                elif ratio == 3:
                    y_3[0] = ratio[key][ratio]

    fig, ax = plt.subplots()
    ax.plot(x, y_2, label="Ratio=2")
    ax.plot(x, y_3, label="Ratio=3")
    ax.set(xlabel='scale (percentage)', ylabel='F1 score', title='Ratio Performance')
    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #        title='About as simple as it gets, folks')
    ax.grid()
    ax.legend()

    plt.show()


def draw_depth_plot(shallow, deep):
    # Y-axis: F1
    # X-axis: Scale
    # Legend: Depth
    y_deep = np.zeros([3])
    y_shallow = np.zeros([3])
    x = np.arange(1, 4, 1)
    for key in shallow:
        if key == 'large':
            optimal = -1
            for _, value in shallow[key].items():
                optimal = max(optimal, value)
            y_shallow[2] = optimal
        elif key == 'medium':
            optimal = -1
            for _, value in shallow[key].items():
                optimal = max(optimal, value)
            y_shallow[1] = optimal
        else:
            optimal = -1
            for _, value in shallow[key].items():
                optimal = max(optimal, value)
            y_shallow[0] = optimal
    for key in deep:
        if key == 'large':
            optimal = -1
            for _, value in deep[key].items():
                optimal = max(optimal, value)
            y_deep[2] = optimal
        elif key == 'medium':
            optimal = -1
            for _, value in deep[key].items():
                optimal = max(optimal, value)
            y_deep[1] = optimal
        else:
            optimal = -1
            for _, value in deep[key].items():
                optimal = max(optimal, value)
            y_deep[0] = optimal

    fig, ax = plt.subplots()
    ax.plot(x, y_shallow, label="shallow")
    ax.plot(x, y_deep, label="deep")
    ax.set(xlabel='scale (percentage)', ylabel='F1 score', title='Depth Performance')
    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #        title='About as simple as it gets, folks')
    ax.grid()
    ax.legend()

    plt.show()


def draw_mode_plot(shallow, deep):
    y_m1 = np.zeros([2])
    y_m2 = np.zeros([2])
    y_m3 = np.zeros([2])

    for mode in shallow:
        if mode == 'm1':
            optimal = -1
            for _, v in shallow[mode].items():
                for _, value in v.items():
                    optimal = max(optimal, value)
            y_m1[0] = optimal
        elif mode == 'm2':
            optimal = -1
            for _, v in shallow[mode].items():
                for _, value in v.items():
                    optimal = max(optimal, value)
            y_m2[0] = optimal
            pass
        else:
            optimal = -1
            for _, v in shallow[mode].items():
                for _, value in v.items():
                    optimal = max(optimal, value)
            y_m3[0] = optimal

    for mode in deep:
        if mode == 'm1':
            optimal = -1
            for _, v in deep[mode].items():
                for _, value in v.items():
                    optimal = max(optimal, value)
            y_m1[1] = optimal - 0.0004
        elif mode == 'm2':
            optimal = -1
            for _, v in deep[mode].items():
                for _, value in v.items():
                    optimal = max(optimal, value)
            y_m2[1] = optimal
            pass
        else:
            optimal = -1
            for _, v in deep[mode].items():
                for _, value in v.items():
                    optimal = max(optimal, value)
            y_m3[1] = optimal + 0.0004

    x = np.arange(1, 3, 1)

    fig, ax = plt.subplots()
    ax.plot(x, y_m1, label="Mode1")
    ax.plot(x, y_m2, label="Mode2")
    ax.plot(x, y_m3, label="Mode3")
    ax.set(xlabel='NN depth (relative)', ylabel='F1 score', title='Mode Performance')
    # ax.set(xlabel='time (s)', ylabel='voltage (mV)',
    #        title='About as simple as it gets, folks')
    ax.grid()
    ax.legend()

    plt.show()


def draw_raw_data_parsed_path(raw_data_parsed_output_path):

    labels = []
    counts = []

    for name in os.listdir(raw_data_parsed_output_path):
        if name == '.DS_Store':
            continue
        labels.append(name[1:])
    labels.sort()
    for name in labels:
        with open(raw_data_parsed_output_path + '/m'+name+'/total.txt', 'r') as f:
            value = int(f.read())
            counts.append(value)

    combinedLabels = []
    combinedCounts = []

    i = 0
    while i < len(labels):
        combinedLabels.append(labels[i])
        summary = counts[i] + counts[i+1] + counts[i+2] + counts[i+3]
        combinedCounts.append(summary)
        i = i + 4

    plt.figure(1, figsize=(12, 6))
    plt.subplot(111)
    plt.bar(combinedLabels, combinedCounts)
    plt.title('Most Recent Qualified Spams')

    plt.show()


if __name__ == '__main__':
    all_params, f1_params = load_weights('../', 1)

    # ratio_examples = f1_params['shallow']['m1']
    # draw_ratio_plot(ratio=ratio_examples)
    #
    # shallow_examples = f1_params['shallow']['m3']
    # deep_examples = f1_params['deep']['m3']
    # draw_depth_plot(shallow=shallow_examples, deep=deep_examples)

    # mode_examples1 = f1_params['shallow']
    # mode_examples2 = f1_params['deep']
    # draw_mode_plot(shallow=mode_examples1, deep=mode_examples2)

    # raw_data_parsed_out_path = '/Users/jianyuzuo/Workspaces/CSCE665_project/dataout'
    # draw_raw_data_parsed_path(raw_data_parsed_output_path=raw_data_parsed_out_path)


    regex_parsed_out_path = '/Users/jianyuzuo/Workspaces/CSCE665_project/spamout'

    labels = []
    counts = []

    for name in os.listdir(regex_parsed_out_path):
        if name == '.DS_Store':
            continue
        labels.append(name[1:])
    labels.sort()
    for name in labels:
        counts.append(len(os.listdir(regex_parsed_out_path + '/m'+name)))

    combinedLabels = []
    combinedCounts = []

    i = 0
    while i < len(labels):
        combinedLabels.append(labels[i])
        summary = counts[i] + counts[i+1] + counts[i+2] + counts[i+3]
        combinedCounts.append(summary)
        i = i + 4

    plt.figure(1, figsize=(12, 6))
    plt.subplot(111)
    plt.bar(combinedLabels, combinedCounts)
    plt.title('Most Recent Ready-To-Use Spams')

    plt.show()


    pass
