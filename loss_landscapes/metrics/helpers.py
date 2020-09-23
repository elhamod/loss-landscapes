import numpy as np
from sklearn.decomposition import PCA
import os
import pickle
import sys
import time
import math
import copy

from loss_landscapes.model_interface.model_parameters import ModelParameters, numpy_to_ModelParameters
from loss_landscapes.model_interface.model_wrapper import wrap_model


# converts  ModelParameters[] to numpy 
def get_model_params_as_numpy(example_vectors):
    return np.concatenate([np.reshape(example_vector.as_numpy(), (-1, 1)) for example_vector in example_vectors], axis=1)

def get_centroid_of_points(example_vectors):
    # get sum
    example_vectors_np = get_model_params_as_numpy(example_vectors)
    sum = np.sum(example_vectors_np, axis=1)
    return np.reshape(sum/ example_vectors_np.shape[0], (-1, 1))

# gives the normal vector to best fitting plane of models.
def get_best_normal_vector_to_models(example_vectors):
    # flatten
    example_vectors_adjusted = get_model_params_as_numpy(example_vectors)
    centroid = get_centroid_of_points(example_vectors)
    example_vectors_adjusted = example_vectors_adjusted - centroid
    svd = np.transpose(np.linalg.svd(example_vectors_adjusted, full_matrices=False)) # full_matrices: Prohibitive memory usage.
    return np.reshape(svd[0][:,-1], (-1, 1))

def get_two_orthognal_bases_from_norm(norm):
    x = np.reshape(np.random.randn(norm.shape[0]), (-1, 1))  # take a random vector
    x -= x.T.dot(norm) * norm       # make it orthogonal to k
    x /= np.linalg.norm(x)  # normalize it to obtain the 2nd one:
    # y = np.cross(norm.T, x.T)      # cross product with k: only works for 2 day
    y = find_orth(x, norm) # prohibitively expensive.
    return x, y

# from https://stackoverflow.com/questions/50660389/generate-a-vector-that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension
from numpy.linalg import lstsq
from scipy.linalg import orth
# bases are n*1
def find_orth(base1, base2):
    O = np.concatenate([base1, base2], axis=1)
    rand_vec = np.random.rand(O.shape[0], 1)
    A = np.hstack((O, rand_vec))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1
    return lstsq(A.T, b)[0]




# Public
def get_best_fit_orthogonal_bases(models):
    return(get_two_orthognal_bases_from_norm(get_best_normal_vector_to_models(models)))

def get_best_fit_orthogonal_bases_pca(models):
    wrapped_model_params = []
    for i in models:
        wrapped_model_params.append(wrap_model(i).get_module_parameters())
    wrapped_model_params_numpy = get_model_params_as_numpy(wrapped_model_params)
    pca = PCA(n_components=2)
    start = time.time()
    pca.fit(wrapped_model_params_numpy.T)
    print("PCA time: ", time.time() - start)

    return numpy_to_ModelParameters(pca.components_[0, :], wrapped_model_params[0]), numpy_to_ModelParameters(pca.components_[1, :], wrapped_model_params[0])


def normalize(model_params, start_point, normalization):
    model_params = copy.deepcopy(model_params)
    start_point = copy.deepcopy(start_point)

    if normalization == 'model':
        model_params.model_normalize_(start_point)
    elif normalization == 'layer':
        model_params.layer_normalize_(start_point)
    elif normalization == 'filter':
        model_params.filter_normalize_(start_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError('Unsupported normalization argument. Supported values are model, layer, and filter')
    return model_params



def project(model_params, dirs_):
    model_params_x = model_params.dot(dirs_[0])/ dirs_[0].model_norm()
    model_params_y = model_params.dot(dirs_[1])/ dirs_[1].model_norm()

    return math.pow((math.pow(model_params_x,2) + math.pow(model_params_y, 2)), 0.5)

def get_non_orth_projections(dirs_, model_params):
    dir1 = dirs_[0]/dirs_[0].model_norm()
    dir2 = dirs_[1]/dirs_[1].model_norm()

    a = np.array([[dir1.dot(dir1),dir1.dot(dir2)], [dir2.dot(dir1),dir2.dot(dir2)]])
    print(a)
    b = np.array([model_params.dot(dir1),model_params.dot(dir2)]) 
    print(b)
    x = np.linalg.solve(a, b)

    return x[0], x[1]



# get best distance for models:
def get_optimal_distance(models, model_start, normalization):#, dirs_):
    distance = None
    
    model_start_wrapper = wrap_model(copy.deepcopy(model_start))
    model_start_params = model_start_wrapper.get_module_parameters()
    model_start_norm = model_start_params.model_norm()
    # print(model_start_params.as_numpy())
    model_start_params_normalized = normalize(model_start_params, model_start_params, normalization)
    # print(model_start_params_normalized.as_numpy())

    print('C', model_start_norm)
    margin_Factor = 1 #TODO: maybe make this bigger?
    # Adjust the distance to encompass all models
    for model in models:
        wrapper_model = wrap_model(copy.deepcopy(model))
       

        # print(wrapper_model.get_module_parameters().as_numpy())
        wrapper_model_normalized = normalize(wrapper_model.get_module_parameters(), model_start_params, normalization)
        # print(wrapper_model_normalized.as_numpy())

        diff =wrapper_model_normalized -model_start_params_normalized
        # print('before diff', diff.model_norm())

        # diff_norm = project(diff, dirs_)

        diff_norm = diff.model_norm()
        print('diff_norm', diff_norm)
        distance_ = diff_norm*2/model_start_norm
        if distance == None or distance < distance_:
            distance = distance_ #*2
    return distance*margin_Factor



# keeps track of the grid
class Coordinates_tracker():
    # dirs_ is a two bases vector list
    def __init__(self,path=None, dirs_=None, dist_=None, steps_=None, scaled_dirs_=None):
        self.dirs_ = dirs_
        self.scaled_dirs_ = scaled_dirs_
        self.dist_ = dist_
        self.steps_ = steps_
        self.save_path = path

    def load(self):
        if os.path.exists(self.save_path):
            try:
                self.dirs_ = pickle.load(open(os.path.join(self.save_path, 'directions'), "rb"))
                self.scaled_dirs_ = pickle.load(open(os.path.join(self.save_path, 'scaled_directions'), "rb"))
                self.dist_ = pickle.load(open(os.path.join(self.save_path, 'distance'), "rb"))
                self.steps_ = pickle.load(open(os.path.join(self.save_path, 'steps'), "rb"))
                return True
            except:
                print("Unexpected error:", sys.exc_info()[0])
                pass

        return False

    def update_distance(self,dist_):
        self.dist_ = dist_
        self.save()

    def update_directions(self,dirs_):
        self.dirs_ = dirs_
        self.save()
    
    def update_scaled_directions(self,scaled_dirs_):
        self.scaled_dirs_ = scaled_dirs_
        self.save()

    def update_steps(self,steps_):
        self.steps_ = steps_
        self.save()
    
    def save(self):
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            pickle.dump(self.dirs_, open( os.path.join(self.save_path, 'directions'), "wb" ))
            pickle.dump(self.dist_, open( os.path.join(self.save_path, 'distance'), "wb"))
            pickle.dump(self.steps_, open( os.path.join(self.save_path, 'steps'), "wb"))
            pickle.dump(self.scaled_dirs_, open( os.path.join(self.save_path, 'scaled_directions'), "wb"))
            return True

        return False