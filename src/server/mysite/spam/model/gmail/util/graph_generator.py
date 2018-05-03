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
                        if mode_str == '1' and depth_str=='deep':
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


if __name__ == '__main__':
    all_params, f1_params = load_weights('../', 1)

    # ratio_examples = f1_params['shallow']['m1']
    # draw_ratio_plot(ratio=ratio_examples)
    #
    # shallow_examples = f1_params['shallow']['m3']
    # deep_examples = f1_params['deep']['m3']
    # draw_depth_plot(shallow=shallow_examples, deep=deep_examples)




    print('End')
    pass
