import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from HEAL.Hyperparameter_optimisation import Hyperopt_train
import numpy as np


BATCH_LIST = [4, 8, 16]
STEP_SIZE_LIST = np.arange(3, 10, dtype=int)

def save_variable(var, filename):
    pickle_f = open(filename, 'wb')
    pickle.dump(var, pickle_f)
    pickle_f.close()
    return filename

space = hp.choice('model_architecture', [
    {
        'Model_name': 'InceptionV3',
        'Learning_rate': hp.uniform('LR1', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma1', 1e-7, 1e-1),
        'Batch_size': hp.choice('Batch_size1', BATCH_LIST),
        'Step_size': hp.choice('Step_size1', STEP_SIZE_LIST),
    },
    {
        'Model_name': 'ResNet50',
        'Learning_rate': hp.uniform('LR2', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma2', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size2', BATCH_LIST),
        'Step_size': hp.choice('Step_size2', STEP_SIZE_LIST),
    },
    {
        'Model_name': 'Vgg16',
        'Learning_rate': hp.uniform('LR3', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma3', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size3', BATCH_LIST),
        'Step_size': hp.choice('Step_size3', STEP_SIZE_LIST),
    },
    {
        'Model_name': 'ShuffleNetV2',
        'Learning_rate': hp.uniform('LR4', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma4', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size4', BATCH_LIST),
        'Step_size': hp.choice('Step_size4', STEP_SIZE_LIST),
    },
    {
        'Model_name': 'MobileNetV2',
        'Learning_rate': hp.uniform('LR5', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma5', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size5', BATCH_LIST),
        'Step_size': hp.choice('Step_size5', STEP_SIZE_LIST),
    },
    {
        'Model_name': 'MNASNET',
        'Learning_rate': hp.uniform('LR6', 1e-7, 1e-1),
        'Gamma': hp.uniform('Gamma6', 0.1, 0.9),
        'Batch_size': hp.choice('Batch_size6', BATCH_LIST),
        'Step_size': hp.choice('Step_size6', STEP_SIZE_LIST),
    },
])


def objective(args):
    Model_name = args['Model_name']
    Learning_rate = args['Learning_rate']
    Gamma = args['Gamma']
    Batch_size = args['Batch_size']
    Step_size = args['Step_size']

    print(args)
    Train_object = Hyperopt_train.HyperoptTrain(Model_name, Learning_rate, Step_size, Gamma, Batch_size)

    return {
        'loss': Train_object.train_model(),
        'status': STATUS_OK}

trials = Trials()

def tuning():
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=30,
                trials=trials)
    print("The best configuration is: " + str(best))
    model_list = ['InceptionV3', 'ResNet50', 'Vgg16', 'ShuffleNetV2', 'MobileNetV2', 'MNASNET']
    model_index = best['model_architecture']
    lr = best["LR" + str(model_index+1)]
    gamma = best["Gamma" + str(model_index+1)]
    batch_size_index = best["Batch_size" + str(model_index+1)]
    step_size_index = best["Step_size" + str(model_index+1)]
    best_config = {'lr': lr,'gamma': gamma,'model_name': model_list[model_index], 'step_size': STEP_SIZE_LIST[step_size_index], 'batch_size': BATCH_LIST[batch_size_index]}
    save_variable(best_config, "HEAL_Workspace/outputs/hyper_parameter.conf")
