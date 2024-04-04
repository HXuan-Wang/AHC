import numpy as np
import torch
import os
import argparse
from FFD import FFD
parser = argparse.ArgumentParser(description='Calculate FFD')

parser.add_argument(
    '--arch',
    type=str,
    default='mobilenet_v1',
    choices=('vgg_16_bn','resnet_56','resnet_50','mobilenet_v1','mobilenet_v2'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='repeat times')

parser.add_argument(
    '--num_layers',
    type=int,
    default=27,
    help='conv layers in the model')

parser.add_argument(
    '--feature_map_dir',
    type=str,
    default='./conv_feature_map',
    help='feature maps dir')
parser.add_argument(
    '--gpu',
    type=str,
    default='4',
    help='Select gpu to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
def reduced_1_row_norm(input, row_index, data_index):

    input[data_index, row_index, :] = torch.zeros(input.shape[-1])

    return input[data_index, :, :]

def ci_score(path_conv):
    conv_output = torch.tensor(np.round(np.load(path_conv), 4)).cuda()
    conv_reshape = conv_output.reshape(conv_output.shape[0], conv_output.shape[1], -1).cuda()

    criterion = FFD().cuda()
    ci = torch.zeros([conv_reshape.shape[0], conv_reshape.shape[1]])
    for i in range(conv_reshape.shape[0]):
        for j in range(conv_reshape.shape[1]):
            input_zero = reduced_1_row_norm(conv_reshape.clone(), j, data_index=i).cuda()
            input_zero = input_zero.unsqueeze(0)
            input_normal = conv_reshape[i, :, :].unsqueeze(0).cuda()
            loss = criterion(input_zero, input_normal)
            ci[i, j] = loss

    # return shape: [batch_size, filter_number]
    return ci

def mean_repeat_ci(repeat, num_layers):
    layer_ci_mean_total = []
    for j in range(num_layers):

        repeat_ci_mean = []
        for i in range(repeat):
            index = j * repeat + i + 1
            path_conv = "./{0}/{1}_repeat5/conv_feature_map_tensor({2}).npy".format(str(args.feature_map_dir),str(args.arch), str(index))
            batch_ci = ci_score(path_conv).cpu().numpy()
            single_repeat_ci_mean = np.mean(batch_ci, axis=0)
            print(single_repeat_ci_mean.shape)
            repeat_ci_mean.append(single_repeat_ci_mean)

        layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
        layer_ci_mean_total.append(layer_ci_mean)
    return np.array(layer_ci_mean_total)



def main():
    repeat = args.repeat

    num_layers = args.num_layers
    save_path = 'FFD_' + args.arch
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ci = mean_repeat_ci(repeat, num_layers)
    for i in range(num_layers):
        print(i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])

if __name__ == '__main__':
    main()