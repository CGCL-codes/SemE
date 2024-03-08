import argparse


parser = argparse.ArgumentParser(description='Run script parameters')

parser.add_argument('--dataset', type=str, nargs='?', default='Aminer_node_cla',
                    help='dataset name')

parser.add_argument('--output_path', type=str, nargs='?', default='Aminer_node_cla_output',
                    help='the path of output files')

parser.add_argument('--processes_num', type=int, nargs='?', default='10',
                    help='the processes number to generate semantic nodes')

parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                    help='GPU_ID (0/1 etc.)')

parser.add_argument('--epochs', type=int, nargs='?', default=100,
                    help='epochs')

parser.add_argument('--batch_size', type=int, nargs='?', default=16,
                    help='Batch size')

parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001,
                    help='Initial learning rate for model.')

parser.add_argument('--weight_decay', type=float, nargs='?', default=5e-4,
                    help='Initial weights decay for model.')

parser.add_argument('--author_num', type=int, nargs='?', default=10206,
                    help='the number of authors or users.')

parser.add_argument('--paper_num', type=int, nargs='?', default=10457,
                    help='the number of papers or businesses.')

parser.add_argument('--conf_num', type=int, nargs='?', default=2584,
                    help='the number of conferences or starts.')

parser.add_argument('--latest_time', type=int, nargs='?', default=1988,
                    help='current time.')

parser.add_argument('--sample_times', type=int, nargs='?', default=8,
                    help='sample times K.')

parser.add_argument('--sample_num', type=int, nargs='?', default=8,
                    help='sample num n.')

parser.add_argument('--train_size', type=float, nargs='?', default=0.6,
                    help='train size of node classification or link prediction')

parser.add_argument('--num_classes', type=int, nargs='?', default=3,
                    help='num of classes')

parser.add_argument("--dim", type=int, nargs='?', default=128,
                    help='embedding dim')

parser.add_argument('--runners', type=int, nargs='?', default=10,
                    help='running times')

parser.add_argument("--dropout", type=float, nargs="?", default="0.5",
                    help="the drop of MLP classify")

args = parser.parse_args()


