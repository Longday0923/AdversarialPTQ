"""
    Train baseline models (AlexNet, VGG, ResNet, and MobileNet)
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import argparse
import numpy as np
from tqdm import tqdm

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# custom
from utils.learner import valid, valid_quantize
from utils.datasets import load_dataset
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer
from utils.qutils import QuantizationEnabler
from utils.jsonutils import _compose_records, _csv_logger, _store_prefix
import json


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_cacc_drop  = 4.                            # accuracy drop thresholds
_best_loss  = 1000.
_quant_bits = [8, 6, 4]                     # used at the validations


# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _run_pgd(epoch, net, dataloader, lossfn,
    use_cuda=False, wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', 
    adv:dict={}):
    accuracies = {}

    # FP model
    cur_facc, cur_floss = valid(
        epoch, net, dataloader, lossfn, adv = adv, use_cuda=use_cuda, silent=True)
    accuracies['32'] = (cur_facc, cur_floss) #TODO update to custom version

    # quantized models
    for each_nbits in _quant_bits:
        cur_qacc, cur_qloss = valid_quantize(
            epoch, net, dataloader, lossfn, adv = adv, use_cuda=use_cuda,
            wqmode=wqmode, aqmode=aqmode, nbits=each_nbits, silent=True)
        accuracies[str(each_nbits)] = (cur_qacc, cur_qloss) #TODO update to custom version
    return accuracies

# ------------------------------------------------------------------------------
#    Training functions
# ------------------------------------------------------------------------------
def run_pgd(parameters):
    global _best_loss


    # init. task name
    task_name = 'eecs598_pgd'


    # initialize the random seeds
    random.seed(parameters['system']['seed'])
    np.random.seed(parameters['system']['seed'])
    torch.manual_seed(parameters['system']['seed'])
    if parameters['system']['cuda']:
        torch.cuda.manual_seed(parameters['system']['seed'])


    # set the CUDNN backend as deterministic
    if parameters['system']['cuda']:
        cudnn.deterministic = True


    # initialize dataset (train/test)
    kwargs = {
            'num_workers': parameters['system']['num-workers'],
            'pin_memory' : parameters['system']['pin-memory']
        } if parameters['system']['cuda'] else {}

    train_loader, valid_loader = load_dataset( \
        parameters['model']['dataset'], parameters['params']['batch-size'], \
        parameters['model']['datnorm'], kwargs)
    print (' : load the dataset - {} (norm: {})'.format( \
        parameters['model']['dataset'], parameters['model']['datnorm']))


    # initialize the networks
    net = load_network(parameters['model']['dataset'],
                       parameters['model']['network'],
                       parameters['model']['classes'])
    if parameters['model']['trained']:
        load_trained_network(net, \
                             parameters['system']['cuda'], \
                             parameters['model']['trained'])
    netname = type(net).__name__
    if parameters['system']['cuda']: net.cuda()
    print (' : load network - {}'.format(parameters['model']['network']))


    # init. loss function
    task_loss = load_lossfn(parameters['model']['lossfunc'])


    # init. optimizer
    optimizer, scheduler = load_optimizer(net.parameters(), parameters)
    print (' : load loss - {} / optim - {}'.format( \
        parameters['model']['lossfunc'], parameters['model']['optimizer']))


    # init. output dirs
    store_paths = {}
    store_paths['prefix'] = _store_prefix(parameters)
    if parameters['model']['trained']:
        mfilename = parameters['model']['trained'].split('/')[-1].replace('.pth', '')
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], task_name, mfilename)
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], task_name, mfilename)
    else:
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])

    # create dirs if not exists
    if not os.path.isdir(store_paths['model']): os.makedirs(store_paths['model'])
    if not os.path.isdir(store_paths['result']): os.makedirs(store_paths['result'])
    print (' : set the store locations')
    print ('  - model : {}'.format(store_paths['model']))
    print ('  - result: {}'.format(store_paths['result']))


    """
        Store the baseline acc.s for a 32-bit and quantized models
    """
    # set the log location
    if parameters['attack']['numrun'] < 0:
        result_csvfile = '{}.csv'.format(store_paths['prefix'])
    else:
        result_csvfile = '{}.{}.csv'.format( \
            store_paths['prefix'], parameters['attack']['numrun'])

    # create a folder
    result_csvpath = os.path.join(store_paths['result'], result_csvfile)
    if os.path.exists(result_csvpath): os.remove(result_csvpath)
    print (' : store logs to [{}]'.format(result_csvpath))

    # compute the baseline acc. for the FP32 model
    base_facc, _ = valid( \
        'Base', net, valid_loader, task_loss, \
        use_cuda=parameters['system']['cuda'], silent=True) #TODO change to custom valid function


    """
        Run the attacks
    """
    # loop over the epochs
    for epoch in range(1, parameters['params']['epoch']+1):

        # : validate with fp model and q-model
        cur_acc_loss = _run_pgd(
            epoch, net, valid_loader, task_loss,
            adv = parameters['adv_attack'],
            use_cuda=parameters['system']['cuda'],
            wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'])
        cur_facc     = cur_acc_loss['32'][0]

        # record the result to a csv file
        cur_labels, cur_valow, cur_vlrow = _compose_records(epoch, cur_acc_loss)
        if not epoch: _csv_logger(cur_labels, result_csvpath)
        _csv_logger(cur_valow, result_csvpath)
        _csv_logger(cur_vlrow, result_csvpath)

    # end for epoch...

    print (' : done.')
    # Fin.


# ------------------------------------------------------------------------------
#    Execution functions
# ------------------------------------------------------------------------------
def dump_arguments(arguments):
    parameters = dict()
    # load the system parameters
    parameters['system'] = {}
    parameters['system']['seed'] = arguments.seed
    parameters['system']['cuda'] = (not arguments.no_cuda and torch.cuda.is_available())
    parameters['system']['num-workers'] = arguments.num_workers
    parameters['system']['pin-memory'] = arguments.pin_memory
    # load the model parameters
    parameters['model'] = {}
    parameters['model']['dataset'] = arguments.dataset
    parameters['model']['datnorm'] = arguments.datnorm
    parameters['model']['network'] = arguments.network
    parameters['model']['trained'] = arguments.trained
    parameters['model']['lossfunc'] = arguments.lossfunc
    parameters['model']['optimizer'] = arguments.optimizer
    parameters['model']['classes'] = arguments.classes
    parameters['model']['w-qmode'] = arguments.w_qmode
    parameters['model']['a-qmode'] = arguments.a_qmode
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    parameters['params']['epoch'] = arguments.epoch
    parameters['params']['lr'] = arguments.lr
    parameters['params']['momentum'] = arguments.momentum
    parameters['params']['step'] = arguments.step
    parameters['params']['gamma'] = arguments.gamma
    # load attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['numbit'] = arguments.numbit
    parameters['attack']['lratio'] = arguments.lratio
    parameters['attack']['margin'] = arguments.margin
    parameters['attack']['numrun'] = arguments.numrun
    # load adversarial attack hyper-parameters
    # type tar num_steps, step_size, epsilon
    parameters['adv_attack'] = {}
    if arguments.att_type is not None:
        parameters['adv_attack']['type'] = arguments.att_type
        parameters['adv_attack']['tar'] = arguments.att_tar
        parameters['adv_attack']['kwargs'] = {}
        parameters['adv_attack']['kwargs']['step_size'] = arguments.att_step_size
        if arguments.att_num_steps is not None:
            parameters['adv_attack']['kwargs']['num_steps'] = arguments.att_num_steps
        if arguments.att_epsilon is not None:
            parameters['adv_attack']['kwargs']['epsilon'] = arguments.att_epsilon
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Run the indiscriminate attack
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the indiscriminate attack')

    # system parameters
    parser.add_argument('--seed', type=int, default=815,
                        help='random seed (default: 215)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers (default: 8)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory')

    # model parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used to train: cifar10.')
    parser.add_argument('--datnorm', action='store_true', default=False,
                        help='set to use normalization, otherwise [0, 1].')
    parser.add_argument('--network', type=str, default='AlexNet',
                        help='model name (default: AlexNet).')
    parser.add_argument('--trained', type=str, default='',
                        help='pre-trained model filepath.')
    parser.add_argument('--lossfunc', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes in the dataset (ex. 10 in CIFAR10).')
    parser.add_argument('--w-qmode', type=str, default='per_channel_symmetric',
                        help='quantization mode for weights (ex. per_layer_symmetric).')
    parser.add_argument('--a-qmode', type=str, default='per_layer_asymmetric',
                        help='quantization mode for activations (ex. per_layer_symmetric).')

    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs to train/re-train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer used to train (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='SGD momentum (default: 0.1)')
    parser.add_argument('--step', type=int, default=0.,
                        help='steps to take the lr adjustments (multiple values)')
    parser.add_argument('--gamma', type=float, default=0.,
                        help='gammas applied in the adjustment steps (multiple values)')

    # attack hyper-parameters
    parser.add_argument('--numbit', type=int, nargs='+',
                        help='the list quantization bits, we consider in our objective (default: 8 - 8-bits)')
    parser.add_argument('--lratio', type=float, default=1.0,
                        help='a constant, the ratio between the two losses (default: 0.2)')
    parser.add_argument('--margin', type=float, default=5.0,
                        help='a constant, the margin for the quantized loss (default: 5.0)')
    
    # adversarial attack hyper=param
    parser.add_argument('--att-type', type=str, default=None)
    parser.add_argument('--att-tar', type=str, default=None)
    parser.add_argument('--att-step-size', type=float, default=None)
    parser.add_argument('--att-num-steps', type=int, default=None)
    parser.add_argument('--att-epsilon', type=float, default=None)

    # for analysis
    parser.add_argument('--numrun', type=int, default=-1,
                        help='the number of runs, for running multiple times (default: -1)')


    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_pgd(parameters)

    # done.