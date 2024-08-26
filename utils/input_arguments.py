import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training multi-class classifier')
    parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                        help='Model architecture')
    parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                        help='Testing Dataset')
    parser.add_argument('--method', type=str,
                        default='ours_c',
                        choices=['rollout', 'lrp', 'partial_lrp', 'transformer_attribution', 'attn_last_layer',
                                 'attn_gradcam', 'generic_attribution', 'ours', 'ours_c'],
                        help='')
    parser.add_argument('--thr', type=float, default=0.,
                        help='threshold')
    parser.add_argument('--start_layer', type=int, default=4,
                        help='start_layer')
    parser.add_argument('--K', type=int, default=1,
                        help='new - top K results')
    parser.add_argument('--save-img', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--no-ia', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fgx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    parser.add_argument('--len-lim', type=int,
                        default=100,
                        help='')
    parser.add_argument('--imagenet-seg-path', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default='models/pretrained_model/')

    return parser
