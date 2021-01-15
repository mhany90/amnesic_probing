from amnesic_probing.debias.debias import debias_by_specific_directions
import json
import numpy as np
import sys
import os

def create_rand_dir_projection(dim, n_coord):
    # creating random directions (vectors) within the range of -0.5 : 0.5
    rand_directions = [np.random.rand(1, dim) - 0.5 for _ in range(n_coord)]

    # finding the null-space of random directions
    rand_direction_projection = debias_by_specific_directions(rand_directions, dim)
    return rand_direction_projection

def get_random_projection(n_coord, vecs_dim=768):
    projection = create_rand_dir_projection(vecs_dim, n_coord)
    return projection

dir_path = sys.argv[1]

# all original Ps
matching_files = [f for f in os.listdir(dir_path) if 'P' in f and 'random' not in f]

meta_file = open(dir_path + '/meta.json', 'r')
meta = json.load(meta_file)
max_n_coord = int(meta['removed_directions'])
n_classes = int(meta['n_classes'])

for p_file in matching_files:
    random_file_name = p_file.replace('.npy', '') + 'random.npy'
    out_file_path = os.path.join(dir_path, random_file_name)
    ## make random proj
    #get n_coords to remove
    p = int(p_file.split('_')[1].replace('.npy', ''))
    n_coord = max_n_coord / p
    random_p = get_random_projection(n_coord)
    np.save(out_file_path, p)


