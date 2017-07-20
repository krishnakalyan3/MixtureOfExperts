

def get_sys_spec():
    import multiprocessing as mp
    from psutil import virtual_memory
    cpu = mp.cpu_count()
    ram = virtual_memory().total / (1048 * (10 ** 6))
    #print("CPU", str(cpu))
    #print("RAM GB", str(ram))
    return (cpu, ram)


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def scale_model(X):
    from sklearn import preprocessing
    scale_d = preprocessing.StandardScaler()
    return scale_d.fit(X)


def load_array(fname):
    import bcolz
    return bcolz.open(fname)[:]


def max_model(X):
    return X.max(axis=0)


def max_transform(max_val, X):
    max_val[max_val == 0] = 1
    return X/max_val


def save_np(file_name, arr):
    import bcolz
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()


def eval_target(y_hat, y):
    from sklearn.metrics import accuracy_score
    y_hat = ohe_decode(y_hat)
    y = ohe_decode(y)
    #cm = confusion_matrix(y, y_hat)
    acc = accuracy_score(y, y_hat)
    error = (1 - acc) * 100
    return error


def comparison(y, y_hat):
    import numpy as np
    y_hat = ohe_decode(y_hat)
    y = ohe_decode(y)
    return np.equal(y, y_hat)

def ohe_decode(y):
    import numpy as np
    y = np.argsort(-1 * y)
    return y[:, 0]


def save_model(model, name):
    from tmp import sys
    """
    Saves a Keras model to disk as two files: a .json with description of the
    architecture and a .h5 with model weights
    Reference: http://keras.io/faq/#how-can-i-save-a-keras-model
    Parameteres:
    ------------
    model: Keras model that needs to be saved to disk
    name: Name of the model contained in the file names:
        <name>_architecture.json
        <name>_weights.h5
    Returns:
    --------
    True: Completed successfully
    False: Error while saving. The function will print the error.
    """
    try:
        # Uses 'with' to ensure that file is closed properly
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        # Uses overwrite to avoid confirmation prompt
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True  # Save successful
    except:
        print(sys.exc_info())  # Prints exceptions
        return False  # Save failed

def load_model(name):
    from keras.models import model_from_json
    """
    Loads a Keras model from disk. The model should be contained in two files:
    a .json with description of the architecture and a .h5 with model weights.
    See save_model() to save the model.
    Reference: http://keras.io/faq/#how-can-i-save-a-keras-model
    Parameters:
    -----------
    name: Name of the model contained in the file names:
        <name>_architecture.json
        <name>_weights.h5
    Returns:
    --------
    model: Keras model object.
    """
    # Uses 'with' to ensure that file is closed properly
    with open(name + '_architecture.json') as f:
        model = model_from_json(f.read())
    model.load_weights(name + '_weights.h5')
    return model


def save_targets(split_buckets):
    import numpy as np
    sd = {}
    for k, v in split_buckets.items():
        for val in v:
            sd[val] = k
    expert_array = []
    for k1 in sorted(sd):
        expert_array.append(sd[k1])

    np.savetxt('expert_vis' + '.csv', expert_array, delimiter=",")
    return 0


def write_csv(data, name):
    import numpy as np
    np.savetxt(name + '.csv', data, delimiter=",")


def config_writer(write_path, config_dict):
    import json
    with open(write_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4)


def config_reader(read_path):
    import json
    with open(read_path, 'r') as f:
        conf_file = json.load(f)
    return conf_file