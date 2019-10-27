import argparse
import time
import logging
import sys
import os
import pickle

sys.path.append(os.path.expanduser('~/darts/cnn'))
from train_class import Train

"""
train arch runner is used in run_experiments_parallel

 - loads data by opening a pickle file containing an architecture spec
 - trains that architecture for e epochs
 - outputs a new pickle file with the architecture spec and its validation loss
"""

def run(args):

    untrained_filepath = os.path.expanduser(args.untrained_filepath)
    trained_filepath = os.path.expanduser(args.trained_filepath)
    epochs = args.epochs
    gpu = args.gpu
    train_portion = args.train_portion
    seed = args.seed
    save = args.save

    # load the arch spec that will be trained
    dic = pickle.load(open(untrained_filepath, 'rb'))
    arch = dic['spec']
    print('loaded arch', arch)

    # train the arch
    trainer = Train()
    val_accs, test_accs = trainer.main(arch, 
                                        epochs=epochs, 
                                        gpu=gpu, 
                                        train_portion=train_portion, 
                                        seed=seed, 
                                        save=save)

    val_sum = 0
    for epoch, val_acc in val_accs:
        key = 'val_loss_' + str(epoch)
        dic[key] = 100 - val_acc
        val_sum += dic[key]
    for epoch, test_acc in test_accs:
        key = 'test_loss_' + str(epoch)
        dic[key] = 100 - test_acc

    val_loss_avg = val_sum / len(val_accs)

    dic['val_loss_avg'] = val_loss_avg
    dic['val_loss'] = 100 - val_accs[-1][-1]
    dic['test_loss'] = 100 - test_accs[-1][-1]
    dic['filepath'] = args.trained_filepath

    print('arch {}'.format(arch))
    print('val loss: {}'.format(dic['val_loss']))
    print('test loss: {}'.format(dic['test_loss']))
    print('val loss avg: {}'.format(dic['val_loss_avg']))

    with open(trained_filepath, 'wb') as f:
        pickle.dump(dic, f)

def main(args):

    #set up save dir
    save_dir = './'

    #set up logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for training a darts arch')
    parser.add_argument('--untrained_filepath', type=str, default='darts_test/untrained_spec_0.pkl', help='name of input files')
    parser.add_argument('--trained_filepath', type=str, default='darts_test/trained_spec_0.pkl', help='name of output files')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--train_portion', type=float, default=0.7, help='portion of training data used for training')
    parser.add_argument('--seed', type=float, default=0, help='random seed to use')
    parser.add_argument('--save', type=str, default='EXP', help='directory to save to')

    args = parser.parse_args()
    main(args)
