import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
import os
import pickle
import sys
import time
import math
import copy
# import torch

import loss_landscapes
from loss_landscapes.model_interface.model_parameters import ModelParameters, numpy_to_ModelParameters
from loss_landscapes.model_interface.model_wrapper import wrap_model


# converts  ModelParameters[] to numpy 
def get_model_params_as_numpy(example_vectors):
    return np.concatenate([np.reshape(example_vector.as_numpy(), (-1, 1)) for example_vector in example_vectors], axis=1)

# def get_centroid_of_points(model_list):
#     # get sum
#     wrapped_model_params = []
#     for i in model_list:
#         wrapped_model_params.append(wrap_model(i).get_module_parameters())
#     example_vectors_np = get_model_params_as_numpy(wrapped_model_params)
#     sum = np.sum(example_vectors_np, axis=1)
#     result = np.reshape(sum/ example_vectors_np.shape[1], (-1, 1))
#     return result

# Gives the projection of the model on the plane made of centroid and two dirs.
# def get_point_projection(model_params, centroid_params, dirs):
#     delta_params = model_params - centroid_params
#     diff1, diff2 = get_non_orth_projections(dirs, delta_params)
#     return centroid_params + numpy_to_ModelParameters(diff1*dirs[0].as_numpy()/dirs[0].model_norm() + diff2*dirs[1].as_numpy()/dirs[1].model_norm(), model_params)



# gives the normal vector to best fitting plane of models.
# def get_best_normal_vector_to_models(example_vectors):
#     # flatten
#     example_vectors_adjusted = get_model_params_as_numpy(example_vectors)
#     centroid = get_centroid_of_points(example_vectors)
#     example_vectors_adjusted = example_vectors_adjusted - centroid
#     svd = np.transpose(np.linalg.svd(example_vectors_adjusted, full_matrices=False)) # full_matrices: Prohibitive memory usage.
#     return np.reshape(svd[0][:,-1], (-1, 1))

# def get_two_orthognal_bases_from_norm(norm):
#     x = np.reshape(np.random.randn(norm.shape[0]), (-1, 1))  # take a random vector
#     x -= x.T.dot(norm) * norm       # make it orthogonal to k
#     x /= np.linalg.norm(x)  # normalize it to obtain the 2nd one:
#     # y = np.cross(norm.T, x.T)      # cross product with k: only works for 2 day
#     y = find_orth(x, norm) # prohibitively expensive.
#     return x, y

# from https://stackoverflow.com/questions/50660389/generate-a-vector-that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension
# from numpy.linalg import lstsq
# from scipy.linalg import orth
# # bases are n*1
# def find_orth(base1, base2):
#     O = np.concatenate([base1, base2], axis=1)
#     rand_vec = np.random.rand(O.shape[0], 1)
#     A = np.hstack((O, rand_vec))
#     b = np.zeros(O.shape[1] + 1)
#     b[-1] = 1
#     return lstsq(A.T, b)[0]




# Public
# def get_best_fit_orthogonal_bases(models):
#     return(get_two_orthognal_bases_from_norm(get_best_normal_vector_to_models(models)))

def get_best_fit_orthogonal_bases_pca(models):
    wrapped_model_params = []
    for i in models:
        wrapped_model_params.append(wrap_model(i).get_module_parameters())
    wrapped_model_params_numpy = get_model_params_as_numpy(wrapped_model_params)
    pca = PCA(n_components=2)
    start = time.time()
    pca.fit(wrapped_model_params_numpy.T)
    print("PCA time: ", time.time() - start)
    result = numpy_to_ModelParameters(pca.components_[0, :], wrapped_model_params[0]), numpy_to_ModelParameters(pca.components_[1, :], wrapped_model_params[0])

    return result


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



# def project(model_params, dirs_):
#     model_params_x = model_params.dot(dirs_[0])/ dirs_[0].model_norm()
#     model_params_y = model_params.dot(dirs_[1])/ dirs_[1].model_norm()

#     return math.pow((math.pow(model_params_x,2) + math.pow(model_params_y, 2)), 0.5)

def get_non_orth_projections(dirs_, model_params):
    dir1 = dirs_[0]/dirs_[0].model_norm()
    dir2 = dirs_[1]/dirs_[1].model_norm()

    a = np.array([[dir1.dot(dir1),dir1.dot(dir2)], [dir2.dot(dir1),dir2.dot(dir2)]])
    b = np.array([model_params.dot(dir1),model_params.dot(dir2)]) 
    x = np.linalg.solve(a, b)

    return x[0], x[1]



# get best distance for models:
def get_optimal_distance(models, model_start, normalization): #, dirs_
    distance = None
    
    model_start_wrapper = wrap_model(copy.deepcopy(model_start))
    model_start_params = model_start_wrapper.get_module_parameters()
    model_start_norm = model_start_params.model_norm()

    margin_Factor = 1.2 #TODO: maybe make this bigger?
    # Adjust the distance to encompass all models
    for model in models:
        wrapper_model = wrap_model(copy.deepcopy(model.cuda()))

        diff =wrapper_model.get_module_parameters() -model_start_params

        diff_norm = diff.model_norm()

        distance_ = diff_norm*2/model_start_norm
        if distance == None or distance < distance_:
            distance = distance_ #*2
    return distance*margin_Factor

# Scale all models on the optimization path to the normalized coordinates.
def scale_model(model_params, center_model_params, scaled_dirs_, normalize, DISTANCE, STEPS):
    diff = model_params - center_model_params

    diff_one, diff_two = loss_landscapes.get_non_orth_projections(scaled_dirs_, diff)
    
#     print(diff.as_numpy())
#     h = diff_one*scaled_dirs_[0].as_numpy()/scaled_dirs_[0].model_norm() + diff_two*scaled_dirs_[1].as_numpy()/scaled_dirs_[1].model_norm()
#     print(h)
#     print('---')
#     print('norm', np.linalg.norm(diff.as_numpy() - h))
    
    scaler=1/center_model_params.model_norm()
    adjust = 0.5*DISTANCE/STEPS

    return diff_one*scaler + adjust , diff_two*scaler + adjust, diff




# keeps track of the grid
class Coordinates_tracker():
    # dirs_ is a two bases vector list
    def __init__(self,path=None, dirs_path=None, dirs_=None, dist_=None, steps_=None, scaled_dirs_=None, centroid_=None, device=None):
        self.dirs_ = dirs_
        self.scaled_dirs_ = scaled_dirs_
        self.dist_ = dist_
        self.steps_ = steps_
        self.centroid_ = centroid_
        self.save_path = path
        self.dirs_path = dirs_path
        self.device=device

    def load_dirs(self):
        if os.path.exists(self.dirs_path):
            try:
                self.dirs_ = pickle.load(open(os.path.join(self.dirs_path, 'directions'), "rb"))
                if self.device is not None and (self.dirs_ is not None):
                    self.dirs_ = (self.dirs_[0].cuda(), self.dirs_[1].cuda())
                return True
            except:
                print("Couldn't load directions")
                print("Unexpected error:", sys.exc_info()[1])
                pass

        return False

    def load(self):
        if os.path.exists(self.save_path):
            try:
                self.scaled_dirs_ = pickle.load(open(os.path.join(self.save_path, 'scaled_directions'), "rb"))
                self.dist_ = pickle.load(open(os.path.join(self.save_path, 'distance'), "rb"))
                self.steps_ = pickle.load(open(os.path.join(self.save_path, 'steps'), "rb"))   
                if self.device is not None and (self.scaled_dirs_ is not None):
                    self.scaled_dirs_ = (self.scaled_dirs_[0].cuda(), self.scaled_dirs_[1].cuda())            
                return True
            except:
                print("Couldn't load scaled directions, number of steps, or distance")
                print("Unexpected error:", sys.exc_info()[1])
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
            pickle.dump(self.dirs_, open( os.path.join(self.dirs_path, 'directions'), "wb" ))
            pickle.dump(self.dist_, open( os.path.join(self.save_path, 'distance'), "wb"))
            pickle.dump(self.steps_, open( os.path.join(self.save_path, 'steps'), "wb"))
            pickle.dump(self.scaled_dirs_, open( os.path.join(self.save_path, 'scaled_directions'), "wb"))
            return True

        return False
    
    # def to_cuda(self, a):
    #     if torch.cuda.is_available() and a is not None:
    #         return a.cuda()
    #     return a


def plot_surface_data(loss_data, models, metric, coord_Tracker, title, normalize='filter', maxVal=None, minVal=None, contour_levels=15, every_x_model=10):
    if maxVal is None:
        maxVal = np.amax([loss_data])
    if minVal is None:
        minVal = np.amin([loss_data])
    
    num_rows=1
    num_columns=1
    STEPS = coord_Tracker.steps_
    DISTANCE = coord_Tracker.dist_

    f, axes = plt.subplots(num_rows, num_columns, figsize=(10, 10), dpi= 300,)

    X = np.array([(k - int(STEPS/2))*DISTANCE/STEPS for k in range(STEPS+1)])
    # X = np.array([(k - int(STEPS/2)) for k in range(STEPS+1)])
    Y = X

    # plot contoues and heatmap
    ax = plt.subplot(num_rows, num_columns, 1)
    contours = plt.contour(X, Y, loss_data, contour_levels, colors='black') # 
    plt.clabel(contours, inline=True, fontsize=10)
    plt.imshow(loss_data, extent=[X[0], X[-1], Y[0], Y[-1]], origin='lower') # , cmap='RdGy', alpha=0.5
    plt.pcolor(X, Y, loss_data, vmin=minVal, vmax=maxVal, cmap='Reds')
    plt.colorbar()

    ax.title.set_text(title) 

    # plot optimzation path
    wm_center = loss_landscapes.wrap_model(models[-1].cuda()).get_module_parameters()
    for i in range(len(models)):
        wm_ = loss_landscapes.wrap_model(models[i].cuda())
        wm = wm_.get_module_parameters()
        scaled_model = scale_model(wm, wm_center, coord_Tracker.scaled_dirs_, normalize, DISTANCE, STEPS)
        plt.plot(scaled_model[0], scaled_model[1], marker='o', markersize=4, color="blue" if i==len(models)-1 else "red")
        value = round(metric(wm_).item(),2)
        if i%every_x_model == 0 or i==len(models)-1:
            plt.text(scaled_model[0], scaled_model[1], value, fontdict={'size'   : 15})
    
    return f