import argparse
import warnings

from crslab.config import Config

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='config/crs/mhim/hredial.yaml', help='config file(yaml) path')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='specify GPU id(s) to use, we now support multiple GPUs. Defaults to CPU(-1).')
    parser.add_argument('-sd', '--save_data', action='store_true',
                        help='save processed dataset')
    parser.add_argument('-rd', '--restore_data', action='store_true',
                        help='restore processed dataset')
    parser.add_argument('-ss', '--save_system', action='store_true',
                        help='save trained system')
    parser.add_argument('-rs', '--restore_system', action='store_true',
                        help='restore trained system')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='use valid dataset to debug your system')
    parser.add_argument('-i', '--interact', action='store_true',
                        help='interact with your system instead of training')
    parser.add_argument('-s', '--seed', type=int, default=2020)
    parser.add_argument('-p', '--pretrain', action='store_true', help='use pretrain weights')
    parser.add_argument('-e', '--pretrain_epoch', type=int, default=9999, help='pretrain epoch')
    args, _ = parser.parse_known_args()
    config = Config(args.config, args.gpu, args.debug, args.seed, args.pretrain, args.pretrain_epoch)

    from crslab.quick_start import run_crslab

    run_crslab(config, args.save_data, args.restore_data, args.save_system, args.restore_system, args.interact,
               args.debug)
