import argparse
import os
import yaml




parser = argparse.ArgumentParser()
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--patch_size", type=int, default=12, help="patch size for train")
parser.add_argument("--scale_factor", type=int, default=8, help="4, 8")
parser.add_argument('--model_name', type=str, default='LF-RWKVNet', help="model name")
parser.add_argument("--channels", type=int, default=32, help="channels , embed_dim for transformer —— C")

parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='./pretrain/LF-RWKVNet_5x5_8x_model.pth',
                    help="path for pre model ckpt")
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL')
parser.add_argument('--path_for_train', type=str, default='/mnt/e/dataset(fixed point)/data_for_training/')
parser.add_argument('--path_for_test', type=str, default='/mnt/e/dataset(fixed point)/data_for_test/')
parser.add_argument('--path_for_val', type=str, default='/mnt/e/dataset(fixed point)/data_for_val/')
parser.add_argument('--path_log', type=str, default='./log/')
parser.add_argument('--patch_size_for_test', default=12, type=int, help='patch size')
parser.add_argument('--stride_for_test', default=6, type=int, help='stride')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 0]')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--epoch', default=80, type=int, help='Epoch to run [default: 80]')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=0, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0)

parser.add_argument('--resume', default=True, type=str,
                    help='Path for checkpoint to load and resume')

args = parser.parse_args()


# parameters for test
args.patch_size_for_test = 12
args.stride_for_test = 6
args.minibatch_for_test = 1
