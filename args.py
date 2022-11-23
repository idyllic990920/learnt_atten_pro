import argparse

def set_args():
    parser = argparse.ArgumentParser(description='Grad Attention Fuse')
    parser.add_argument('--cu_num', default='0,1', type=str)
    parser.add_argument('--gpu_id', default='1', type=int)
    parser.add_argument('--seed', default='1', type=int)
    parser.add_argument('--dataset', default='TE', type=str)
    parser.add_argument('--log_name', default='./log/debug', type=str)
    parser.add_argument('--model', default=['CNN1', 'CNN2', 'CNN3', 'CNN4', 'CNN5',
                                            'MLP1', 'MLP2', 'MLP3', 'MLP4', 'MLP5'], type=list)

    # heterogeneous data split
    parser.add_argument('--train_ratio', default='0.7', type=float)
    parser.add_argument("--resplit", type=int, default=0, 
                        help="Whether need to split again (resplit=1) or \
                              just read data already splited (resplit=0) ")
    parser.add_argument("--n_class", type=int, default=15, help="number of classification labels")
    parser.add_argument("--min_sample", type=int, default=100, help="Min number of samples per user.")
    parser.add_argument("--unknown_test", type=int, default=1, 
                        help="Whether allow test label unseen for each user.")
    parser.add_argument("--alpha", type=float, default=0.1, 
                        help="alpha in Dirichelt distribution (smaller means larger heterogeneity)")
    parser.add_argument("--n_user", type=int, default=10,
                        help="number of local clients, should be muitiple of 10.")
    # parser.add_argument("--sampling_ratio", type=float, default=0.05, help="Ratio for sampling training samples.")

    # models training
    parser.add_argument('--lr', default='1e-3', type=float, help='Learning rate of local training')
    parser.add_argument('--global_lr', default='1e-3', type=float, help='Learning rate of global model')
    parser.add_argument('--gen_lr', default='1e-3', type=float, help='Learning rate of generator')
    parser.add_argument('--gamma', default='0.99', type=float, help='Decaying rate')
    parser.add_argument('--decay', default='0', type=int, help='whether decay learning rate on cloud')

    parser.add_argument('--epochs', default='5', type=int)
    parser.add_argument('--global_epoch', default='5', type=int)
    parser.add_argument('--iterations', default='50', type=int)
    parser.add_argument('--n_critic', default='10', type=int)
    parser.add_argument('--batch_size', default='32', type=int)
    parser.add_argument('--batch_size_global', default='32', type=int)

    # hyper parameters
    parser.add_argument('--total_num', default='900', type=int)
    parser.add_argument('--weight_threshold', default='0.5', type=float)
    parser.add_argument('--z_dim', default='128', type=int)
    parser.add_argument('--hid_dim1', default='64', type=int)
    parser.add_argument('--hid_dim2', default='64', type=int)

    # loss coeeficients
    # client training
    # parser.add_argument('--alpha_atten', default='1', type=float, help='local training atten weight')
    parser.add_argument('--beta', default='0.1', type=float, help='local training l2 regularization')
    # generator training
    parser.add_argument('--lamda_proto', default='1', type=float, 
                                         help='coefficient of atten in generator training')
    parser.add_argument('--lamda_atten', default='1', type=float, 
                                         help='coefficient of atten in generator training')
    parser.add_argument('--lamda_label', default='1', type=float, 
                                         help='coefficient of label in generator training')
    parser.add_argument('--lamda_ms', default='0.1', type=float, 
                                         help='coefficient of label in generator training')

    # # global model training
    # parser.add_argument('--beta_proto', default='1', type=float, 
    #                                      help='coefficient of atten in global training')
    # parser.add_argument('--beta_atten', default='1', type=float, 
    #                                      help='coefficient of atten in global training')
    # parser.add_argument('--beta_label', default='1', type=float, 
    #                                      help='coefficient of label in global training')

    # plot hyperparameters 
    parser.add_argument('--plot_num', default='450', type=int)
    parser.add_argument('--plot_batch_size', default='450', type=int)

    args = parser.parse_args()
    print(args)
    return args



