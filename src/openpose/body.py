import cv2
import numpy as np
import math
import time
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib

import jax
from jax import lax, random
import jax.numpy as jnp

from flax.core import freeze, FrozenDict
import flax.linen as nn

import torch

try:
    import torch_xla.core.xla_model as xm
    _xla_available = True
except ImportError:
    _xla_available = False

from . import util
from .model import bodypose_model

#import util
#from model import bodypose_model

import jax.scipy as jsp

@jax.jit
def gaussian_filter(image, sigma, window_size=None):
#    window_size = jnp.int(3*sigma) if window_size is None else window_size
    print(image.shape, sigma)
    window_size = 9
    x = jnp.linspace(-window_size/sigma, window_size/sigma, 2*window_size+1)
    window = jsp.stats.norm.pdf(x) * jsp.stats.norm.pdf(x[:, None])
    return jsp.signal.convolve(image, window[:,:,None], mode='same')



def _get_flax_keys(keys):
  if keys[0] == 'features':
    keys[0] = 'backbone'
  if keys[-1] == 'weight':
    is_scale = 'norm' in keys[-2] if len(keys) < 6 else 'norm' in keys[-3]
    keys[-1] = 'scale' if is_scale else 'kernel'
  if 'running' in keys[-1]:
    keys[-1] = 'mean' if 'mean' in keys[-1] else 'var'
  if keys[-2] in ('1', '2'):  # if index separated from layer name, concatenate
    keys = keys[:3] + [keys[3] + keys[4]] + [keys[5]]
  return keys


def _densenet(rng, arch, growth_rate, block_config, num_init_features, pretrained, **kwargs):
  densenet = DenseNet(growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features, **kwargs)

  if pretrained:
    torch_params = utils.load_torch_params(model_urls[arch])
    flax_params = FrozenDict(utils.torch_to_linen(torch_params, _get_flax_keys))

def torch_to_linen(torch_params, get_flax_keys):
  """Convert PyTorch parameters to Linen nested dictionaries"""

  def add_to_params(params_dict, nested_keys, param, is_conv=False):
    if len(nested_keys) == 1:
      key, = nested_keys
      params_dict[key] = np.transpose(param, (2, 3, 1, 0)) if is_conv else np.transpose(param)
    else:
      assert len(nested_keys) > 1
      first_key = nested_keys[0]
      if first_key not in params_dict:
        params_dict[first_key] = {}
      add_to_params(params_dict[first_key], nested_keys[1:], param, ('conv' in first_key and \
                                                                     nested_keys[-1] != 'bias'))

  flax_params = {'params': {}, 'batch_stats': {}}
  for key, tensor in torch_params.items():
    flax_keys = get_flax_keys(key.split('.'))
    if flax_keys[-1] is not None:
      if flax_keys[-1] in ('mean', 'var'):
        add_to_params(flax_params['batch_stats'], flax_keys, tensor.detach().numpy())
      else:
        add_to_params(flax_params['params'], flax_keys, tensor.detach().numpy())

  return flax_params


@jax.jit
def get_line_score(b, vA, vB, paf_avg, idxA, idxB):
    mid_num=10
    
    vec = jnp.subtract(vB, vA)
    norm = jnp.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
    norm = jnp.maximum(0.001, norm)
    vec = jnp.divide(vec, norm)

    """
    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                      np.linspace(candA[i][1], candB[j][1], num=mid_num)))

    N = mid_num

    vec_x = np.array([paf_avg[b, int(round(startend[I][1])), int(round(startend[I][0])), idxA] \
                    for I in range(N)])
    vec_y = np.array([paf_avg[b, int(round(startend[I][1])), int(round(startend[I][0])), idxB] \
                    for I in range(N)])
    """
    N = mid_num

    mp_i = jnp.linspace(vA[1], vB[1], dtype=jnp.int32, num=mid_num)
    mp_j = jnp.linspace(vA[0], vB[0], dtype=jnp.int32, num=mid_num)

    score_midpts =  vec[0]*paf_avg[b, mp_i, mp_j, idxA] + vec[1]*paf_avg[b, mp_i, mp_j, idxB]
    return norm, score_midpts

class Body(object):
    def __init__(self, model_path):
        model = bodypose_model()

        key1, key2 = random.split(random.PRNGKey(0))
#        x = random.normal(key1, (1,3,512,512,))
        x = random.normal(key1, (8,512,512,3))
        params = model.init(key2, x)        
        torch_params = torch.load(model_path)
           
        params = FrozenDict(torch_to_linen(torch_params, _get_flax_keys))

        @jax.jit
        def get_maps(oriImg):
            scale_search = [0.5]
            boxsize = 368
            stride = 8
            padValue = 128
            thre1 = 0.1


            multiplier = [x * boxsize / oriImg.shape[1] for x in scale_search]
            heatmap_avg = jnp.zeros(oriImg.shape[0:2]+(19,))
            paf_avg = jnp.zeros(oriImg.shape[0:2] + (38,))

            for m in range(len(multiplier)):
                scale = multiplier[m]
                imageToTest = util.smart_resize_k(oriImg, fx=scale, fy=scale)
                imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
                im = imageToTest_padded / 256 - 0.5
                data = im #jnp.array(im, dtype=jnp.float32)

                # data = data.permute([2, 0, 1]).unsqueeze(0).float()
                Mconv7_stage6_L1, Mconv7_stage6_L2 = model.apply(params, data)

                # extract outputs, resize, and remove padding
                # heatmap = np.trangspose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1, 2, 0))  # output 1 is heatmaps
                heatmap = Mconv7_stage6_L2  # output 1 is heatmaps
                heatmap = util.smart_resize_k(heatmap, fx=stride, fy=stride)
                heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
                heatmap = util.smart_resize(heatmap, (oriImg.shape[0], oriImg.shape[1]))

                paf = Mconv7_stage6_L1  # output 0 is PAFs
                paf = util.smart_resize_k(paf, fx=stride, fy=stride)
                paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
                paf = util.smart_resize(paf, (oriImg.shape[0], oriImg.shape[1]))

                heatmap_avg +=  heatmap / len(multiplier)
                paf_avg +=  paf / len(multiplier)


            # Smooth heatmap (on original image scale)
            one_heatmap = gaussian_filter(heatmap_avg, sigma=3)

            max_lr = nn.max_pool(one_heatmap, (3,1), padding="SAME")
            max_ud = nn.max_pool(one_heatmap, (1,3), padding="SAME")

            # Extract local maxima for which smoothed heatmap is above threshold
            peaks_binary = jnp.logical_and(jnp.logical_and( one_heatmap >= max_lr, one_heatmap >=max_ud), one_heatmap > thre1)


            return {'heatmap': heatmap_avg, 'paf': paf_avg, 'peaks': peaks_binary} #[[jnp.nonzero(peaks_binary[b,:,:,p]) for p in range(peaks_binary.shape[3])] for b in range(oriImg.shape[0])]

        self.get_maps = jax.vmap(get_maps)
        
    def __call__(self, oriImg):
        thre2 = 0.05
        bs = oriImg.shape[0]
        print(oriImg.shape)
        res = self.get_maps(oriImg)
        heatmap_avg = res['heatmap']#.astype(jnp.float32)
        paf_avg= res['paf']#.astype(jnp.float32)
        peaks_binary = res['peaks']#.astype(jnp.float32)
        candidates = []
        subsets = []

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]


        
        for b in range(bs):
            all_peaks = []
            peak_counter = 0
            # For each part - find local maxima of smoothed aggregated heatmaps
            # Add Id, score them according to aggregated heatmap.
            for part in range(18):
                map_ori = heatmap_avg[b, :, :, part]
                peaks_list = jnp.nonzero(peaks_binary[b, :, :, part])
                peaks = list(zip(peaks_list[1], peaks_list[0]))  # note reverse

                peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
                peak_id = range(peak_counter, peak_counter + len(peaks))
                peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

                all_peaks.append(peaks_with_score_and_id)
                peak_counter += len(peaks)

            connection_all = []
            special_k = []
            mid_num = 10

            # For each connection in the skeleton, find the potential connections
            for k in range(len(mapIdx)):
                idxA = mapIdx[k][0] - 19
                idxB = mapIdx[k][1] - 19
                candA = all_peaks[limbSeq[k][0] - 1]
                candB = all_peaks[limbSeq[k][1] - 1]
                
                nA = len(candA)
                nB = len(candB)
                indexA, indexB = limbSeq[k]
                if (nA != 0 and nB != 0):
                    connection_candidate = []
                    for i in range(nA):
                        for j in range(nB):

                            # Look at the paf maps along the lines joining the potential peaks
                            vA = jnp.array(candA[i][:2])
                            vB = jnp.array(candB[j][:2])
                            norm, score_midpts = get_line_score(b, vA, vB, paf_avg, idxA, idxB)
                            
                            #score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                            score_with_dist_prior = jnp.sum(score_midpts) / len(score_midpts) + min(
                                0.5 * oriImg.shape[1] / norm - 1, 0)
                            criterion1 = jnp.sum(score_midpts > thre2) > 0.8 * len(score_midpts)
                            criterion2 = score_with_dist_prior > 0
                            if criterion1 and criterion2:
                                connection_candidate.append(
                                    [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                    # 
                    connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                    connection = np.zeros((0, 5))
                    for c in range(len(connection_candidate)):
                        i, j, s = connection_candidate[c][0:3]
                        if (i not in connection[:, 3] and j not in connection[:, 4]):
                            connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                            if (len(connection) >= min(nA, nB)):
                                break

                    connection_all.append(connection)
                else:
                    special_k.append(k)
                    connection_all.append([])

            # last number in each row is the total parts number of that person
            # the second last number in each row is the score of the overall configuration
            subset = -1 * np.ones((0, 20))
            candidate = np.array([item for sublist in all_peaks for item in sublist])

            for k in range(len(mapIdx)):
                if k not in special_k:
                    partAs = connection_all[k][:, 0]
                    partBs = connection_all[k][:, 1]
                    indexA, indexB = np.array(limbSeq[k]) - 1

                    for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                        found = 0
                        subset_idx = [-1, -1]
                        for j in range(len(subset)):  # 1:size(subset,1):
                            if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                                subset_idx[found] = j
                                found += 1

                        if found == 1:
                            j = subset_idx[0]
                            if subset[j][indexB] != partBs[i]:
                                subset[j][indexB] = partBs[i]
                                subset[j][-1] += 1
                                subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                        elif found == 2:  # if found 2 and disjoint, merge them
                            j1, j2 = subset_idx
                            membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                            if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                                subset[j1][:-2] += (subset[j2][:-2] + 1)
                                subset[j1][-2:] += subset[j2][-2:]
                                subset[j1][-2] += connection_all[k][i][2]
                                subset = np.delete(subset, j2, 0)
                            else:  # as like found == 1
                                subset[j1][indexB] = partBs[i]
                                subset[j1][-1] += 1
                                subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                        # if find no partA in the subset, create a new subset
                        elif not found and k < 17:
                            row = -1 * np.ones(20)
                            row[indexA] = partAs[i]
                            row[indexB] = partBs[i]
                            row[-1] = 2
                            row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                            subset = np.vstack([subset, row])
            # delete some rows of subset which has few parts occur
            deleteIdx = []
            for i in range(len(subset)):
                if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                    deleteIdx.append(i)
            subset = np.delete(subset, deleteIdx, axis=0)

            # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
            # candidate: x, y, score, id
            candidates.append(candidate)
            subsets.append(subset)
        return candidates, subsets

if __name__ == "__main__":
    body_estimation = Body('ckpts/body_pose_model.pth')

    test_image = 'dancer.png'
    oriImg = cv2.imread(test_image)  # B,G,R order
    H, W, _ = oriImg.shape
    candidates, subsets = body_estimation(oriImg[None])
    print(candidates, subsets)

    candidate = candidates[0]
    subset = subsets[0]
    if candidate.ndim == 2 and candidate.shape[1] == 4:
        candidate = candidate[:, :2]
        candidate[:, 0] /= float(W)
        candidate[:, 1] /= float(H)
    else:
        raise
        
    canvas = util.draw_bodypose(np.zeros_like(oriImg), candidate, subset)
    plt.imshow(canvas)
    plt.show()
