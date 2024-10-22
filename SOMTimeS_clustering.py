import pandas as pd
import numpy as np
import somtimes
import os
import matplotlib.pyplot as plt
from SelfOrganizingMap import SelfOrganizingMap
from scipy.spatial.distance import cdist
from tslearn import metrics
import copy
from dtaidistance import dtw
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats
from matplotlib.dates import MonthLocator, DateFormatter

# creating a corresponding color list with the cluster assignment
_colors = ['#0cf4f4', '#930d0f', '#f6f918', '#1630ee', '#e259d0', '#139913', '#ff7f00', '#8db0c2', 'olivedrab', 
          'darkorange', 'cadetblue', 'turquoise', 'palegreen', 'goldenrod','darkred', 'coral', 'purple', 'navy']

global _colors

### Functions ###
# creating a corresponding color list with the cluster assignment
_colors = ['#0cf4f4', '#930d0f', '#f6f918', '#1630ee', '#e259d0', '#139913', '#ff7f00', '#8db0c2', 'olivedrab', 
          'darkorange', 'cadetblue', 'turquoise', 'palegreen', 'goldenrod','darkred', 'coral', 'purple', 'navy']
#_colors = ['grey', 'red', 'yellow', 'dodgerblue', 'violet', 'olivedrab', 
          # 'darkorange', 'cadetblue', 'turquoise', 'palegreen', 'goldenrod',
           #'darkred', 'coral', 'purple', 'navy']

global _colors


def bmu(x,windowSize,candidateWeights, som_weights):
        ## function to find winning node
        #: input observatiopn

        # format input for use in this function --- dtw distance
        # x = np.reshape(x[0], (1, 1, len(x[0])))


        ####################################
        # calculate distances (in Euclidean and DTW it is the minimum). Iterate over all nodes to find distance
        #x = np.reshape(x,(1,len(x[0])))
        min_distance = float('inf')
        dtw_Cals = 0
        for i in candidateWeights:

            #get candidate weight
            #this needs to be a deep copy for dtw distance not to throw an error.
            weights = copy.deepcopy(som_weights[i])
            xCopy = copy.deepcopy(x)
            #get dtw distance
            distance = dtw.distance_fast(xCopy,weights, window = windowSize+1, use_pruning=True)
            
            dtw_Cals+=1
            #update min distance if new distance is lower
            if distance<min_distance:
                bmuIndex = i
                min_distance = distance
            #update min distance if new distance is equal to min distance
            elif distance==min_distance and np.random.uniform(low=0, high=1) <0.3:
                bmuIndex = i
                min_distance = distance

        return [bmuIndex,dtw_Cals]


def Locate_samples(samples, som_weights, windowSize, hiddenSize=None):

    Eucdistances = cdist(samples, som_weights, 'euclidean')
    upper_bounds = Eucdistances.min(axis=1)
    bmus = []
    dtw_Cals = 0


    for i in range(0,len(samples)):
        # calculate lower bounds
        lower_bounds = []
        for j in range(0, len(som_weights)):
            lower_bounds.append(metrics.lb_keogh(samples[i], som_weights[j], radius=windowSize))

        lower_bounds = np.asarray(lower_bounds)
        #bmu is the min node based on sq euc distance

        #if min Euclidean distance between observation and all neurons is less than or equal to the minimum lower bound then node with minimum euclidean distance is the best matching neuron., 
        if upper_bounds[i]<=min(lower_bounds):
            bmus.append(np.argmin(Eucdistances[i]))


        #lower bounds are greater than bmu based on squared euclidean distance
        else:
            #still no need to calculate dtw with all nodes
            candidateWeights = np.argwhere(lower_bounds < upper_bounds[i]).flatten()

            #get best matching unit
            [_bmu,calls] = bmu(samples[i], windowSize, candidateWeights, som_weights)
            bmus.append(_bmu)
            dtw_Cals +=calls

    _locations = []
    if hiddenSize==None:
        x = int(som_weights.shape[0]**0.5)
        hiddenSize= [x, x]
    for idx in bmus:
        _locations.append(np.unravel_index(idx, hiddenSize))

    _locations = [[x[0],x[1]] for x in _locations]
    for loc in _locations: # shifting x values to be centered
        if loc[1]%2==1:
            loc[0] = loc[0]-0.5

    return _locations


def _somtimes_func(input_arr, _window_size, grid_x, grid_y, niteration=50):

    #np.random.seed(0)
    epochs = niteration

    print('Creating SOM... ')
    hiddenSize = [grid_x, grid_y]
    print('Hidden Size is: '+str(hiddenSize))
    SOM = somtimes.SelfOrganizingMap(inputSize = len(input_arr[0]), hiddenSize = hiddenSize)

    wSize = _window_size
    stats = SOM.iterate(input_arr,epochs = epochs,windowSize = wSize,k=1,randomInitilization=True)

    return SOM


def loc_check(wxx, wyy, w_lst):
    
    yrand = [0.1, -0.1]
    init_wxx = wxx
    init_wyy = wyy
    
    while [1 for i in w_lst if i==(wxx,wyy)]:
        wxx = wxx + np.random.choice(yrand)
        wyy = wyy + np.random.choice(yrand)
        
    w_lst.append((wxx, wyy))
    
    return wxx, wyy, w_lst


def build_lattice(SOM_class):

    x = SOM_class.getWeights().shape[0]**0.5
    _xx, _yy = np.meshgrid(np.arange(x), np.arange(x))
    _xx[::-2] -= 0.5 # for hexagonal topology
    return _xx.T, _yy.T
    

def Umatrix_plot(SOM_class, Samples_df ,window_size, _cmap='gray'):

    Samples = Samples_df.to_numpy()
    print(f'sample size of {Samples.shape}')
    xx, yy = build_lattice(SOM_class)
    sample_locations = Locate_samples(Samples, SOM_class.getWeights(), window_size)
    umatrix = SOM_class.createUmatrix(window_size)
    norm_umatrix = np.rot90(normalize(umatrix,norm='max')[:, ::-1])

    #plt.imshow(umatrix, cmap='Greys',alpha=1)
    
    weights = SOM_class.getWeights().reshape(xx.shape[0], xx.shape[1], Samples.shape[1]) 
    print(f'weight size:{weights.shape}')
    
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)

    ax.set_aspect('equal')
    
    # Normalizing weights (comment this to have between zero and 1 refrectance values in hexagonals)
    # w = (w-np.min(w))/(V_max-np.min(w))

    # iteratively add hexagons
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius=1 / np.sqrt(3),
                                 facecolor=cm.Greys(norm_umatrix[i,j]),
                                 alpha=1.0,
                                 edgecolor='gray')
            ax.add_patch(hex)

    Ws = [(100, 100)] # an arbitarary matrix just to start the loc_check function
    # Superimposing the samples
    for cnt, x in enumerate(Samples):
        
        wx = sample_locations[cnt][0] 
        wy = sample_locations[cnt][1] * np.sqrt(3) / 2

        wx, wy, Ws = loc_check(wx, wy, Ws) # check to avoid overlaying sample points on top of eachother
        
        plt.plot(wx, wy, 
                 'o', 
                 markerfacecolor='None',
                 markeredgecolor='None', 
                 markersize=5, 
                 markeredgewidth=2,
                 alpha=1)
        
    xrange = np.arange(weights.shape[0])
    yrange = np.arange(weights.shape[1])
    plt.xticks(xrange-.5, xrange)
    plt.yticks(yrange * np.sqrt(3) / 2, yrange)
    ax.axis('off')

    
    plt.show()


def poly_colors(_som_weights, number_of_clusters):
    
    _som_grid_x = _som_weights.shape[0] # ncols
    _som_grid_y = _som_weights.shape[1] # nrows
    _som_grid_z = _som_weights.shape[2] # n features
    _som_n_nodes = int(_som_grid_x*_som_grid_y)
    
    # H clustering of weights
    
    linkage_matrix = linkage(_som_weights.reshape(_som_n_nodes,_som_grid_z), method='ward', metric='euclidean')
    print(linkage_matrix.shape)
    linkage_matrix_1 = linkage(_som_weights.reshape(_som_n_nodes,_som_grid_z), method='single', metric=lambda t1,t2: dtw.distance_fast(t1, t2))
    print(linkage_matrix_1.shape)
    cluster_assignments = fcluster(linkage_matrix, t=number_of_clusters, criterion='maxclust')
    
    color_labels = [_colors[val-1] for val in cluster_assignments]
    
    return np.asarray(color_labels).reshape(_som_grid_x, _som_grid_y), cluster_assignments


def get_samples_cluster_number(raw_samples_on_grid_loc, hcluster_assignments):

    x = int(np.sqrt(len(hcluster_assignments))) # dims hardcoded
    samples_cluster_number = []
    for loc in raw_samples_on_grid_loc:
        samples_cluster_number.append(hcluster_assignments[np.ravel_multi_index(loc, dims=(x,x))]) 
        
    return samples_cluster_number


def closest_to_median(samples_2d_nparr):

    data = np.asarray(samples_2d_nparr)
    
    # Calculate the median row
    median_row = np.median(data, axis=0)

    # Calculate the DTW distance between each row and the median row (same method as SOMTimeS)
    distances = [dtw.distance_fast(median_row, np.asarray(row)) for row in data]

    # Find the index of the row with the smallest DTW distance to the median row
    closest_row_index = np.argmin(distances)
    closest_row = data[closest_row_index]

    #return closest_row
    return median_row

    
def TScluster(_st_weights, samples, sample_locations, n_cluster, colored_lattice=False):
    """
    This function receives the somtimes trained weights (2d numpy array),
    samples: time series data in the form of 2d array (each row is a time series)
    locations of samples on the som grid which is a list of tuples
    and returns a plot of time series cluster
    """

    data = _st_weights
    
    # Clustering of weights (nodes) using h clustering
    # seed = 0
    colors_of_nodes, cluster_no_of_nodes = poly_colors(data, number_of_clusters=n_cluster)
    cluster_number_of_samples = get_samples_cluster_number(sample_locations, cluster_no_of_nodes)
    
    # Row and Column calculation based of number of clusters
    if n_cluster>20:
        print('Number of clusters could not be more than 7')
        return
    elif n_cluster%2 == 0:
        num_rows = int(n_cluster/2)
        num_cols = 2 
    #elif n_cluster%3 == 0:
        #num_rows = int(n_cluster/3)
        #num_cols = 3
    #elif n_cluster%5 == 0:
        #num_rows = 3
        #num_cols = 2
    #elif n_cluster%7 == 0:
        #num_rows = 4
        #num_cols = 2
    else:
        num_rows = int(n_cluster/2) + 1
        num_cols = 2
    
    
    # Plotting clusters
    _figsize = (num_cols*5, num_rows*2.2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=_figsize, sharex=True, sharey=True, constrained_layout=True, frameon=False)
    #main_title = settitle(var_name)
    fig.suptitle("", fontsize=20,va='top',ha='center')
    clus_num = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if clus_num >= n_cluster:
                break
            
            samples_in_cluster_x_indices = np.where(np.asarray(cluster_number_of_samples)==clus_num+1)[0].tolist()
            for xx in samples[samples_in_cluster_x_indices, :]:
                axs[r, c].plot(xx, color = 'grey' , alpha=0.08)
            axs[r, c].plot(closest_to_median(samples[samples_in_cluster_x_indices, :]), color = _colors[clus_num])
            Nvalue = len(samples_in_cluster_x_indices)
            axs[r,c].set_title('Cluster '+ str(clus_num+1) + ', Nvalue = '+str(Nvalue),fontsize=15 )
            # axs[r,c].set_xlabel('Day of the year', fontsize=12)
            # axs[r,c].set_ylabel('Rescaled value', fontsize=12)
            #if var_name=='thresh_20_jd_30d_wndw':
                #axs[r,c].set_ylabel('Rescaled value', fontsize=12)
            #else:
                #axs[r,c].set_ylabel('Semivariance value', fontsize=12)
            clus_num += 1 
    if n_cluster%2!=0:
        axs[num_rows-1, num_cols-1].axis('off') 
    fig.text(0.5, -0.05, 'Day of the year', ha='center', fontsize=15)
    fig.text(-0.05, 0.5, 'Normalized 20 percent variable streamflow percentile', va='center', rotation='vertical', fontsize=15)
    
    plt.savefig('timeseries_clusterplots.svg')
    plt.savefig('timeseries_clusterplots_transparent.svg',transparent=True)
    
    # adding another plot that is the color coded grid based on the hclusters
    if colored_lattice:
        f2 = plt.figure(figsize=(10,10))
        ax1 = f2.add_subplot(111)
        ax1.set_aspect('equal')
        # iteratively add hexagons
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                wy = yy[(i, j)] * np.sqrt(3) / 2
                hex = RegularPolygon((xx[(i, j)], wy), 
                                     numVertices=6, 
                                     radius=1 / np.sqrt(3),
                                     facecolor=colors_of_nodes[i,j],
                                     alpha=1.0,
                                     edgecolor='gray')
                ax1.add_patch(hex)


        
    plt.show()  
    
    return colors_of_nodes, cluster_no_of_nodes, cluster_number_of_samples

def getLocations(SOM_class, samples, _wSize):
    
    list_of_dict_loc = SOM_class.locationOnMap(samples, windowSize=_wSize)
    _sample_locations = np.asarray([(d['x'], d['y']) for d in list_of_dict_loc])
    
    #print(_sample_locations)
    return _sample_locations

def plot_colored_grid(_som_class, N_features, color_of_nodes_list, nodes_cluster_no, samples_cluster_no, som_window_size):
    
    Samples = N_features
    xx, yy = build_lattice(_som_class)
    sample_locations = Locate_samples(Samples, _som_class.getWeights(), som_window_size)
    weights_from_som = _som_class.getWeights().reshape(xx.shape[0], xx.shape[1], Samples.shape[1]) 
    print(f'weight size:{weights_from_som.shape}')
    #weights_from_som = _som_class.get_Weights()
    pgcolors = color_of_nodes_list
    hclus_assignments = nodes_cluster_no
    
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    # adding hexagons with colors based on clustered nodes (pgcolors)
    for i in range(weights_from_som.shape[0]):
        for j in range(weights_from_som.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius=.99 / np.sqrt(3),
                                 facecolor=pgcolors[i,j], 
                                 alpha=1, 
                                 edgecolor='white')
            ax.add_patch(hex)

    # Superimposing the samples
    Ws = [(100, 100)] # Some random big tuple (greater than the gird size)
    samples_on_grid_loc = []
    Samples=N_features
    for cnt, x in enumerate(Samples):
        wx = sample_locations[cnt][0] 
        wy = sample_locations[cnt][1] * np.sqrt(3) / 2
        wx, wy, Ws = loc_check(wx, wy, Ws)
    
        plt.plot(wx, wy, 
                 'o', 
                 markerfacecolor='None',
                 markeredgecolor='black', 
                 markersize=4, 
                 markeredgewidth=0.4,
                 alpha=1)
        
    # Plot    
    xrange = np.arange(weights_from_som.shape[0]); yrange = np.arange(weights_from_som.shape[1])
    plt.xticks(xrange-.5, xrange); plt.yticks(yrange * np.sqrt(3) / 2, yrange)
    ax.axis('off')
    plt.show()

    return Ws

def grid_with_colored_dots(_som_class, N_features, samples_loc_on_grid, nodes_cluster_no, target_labels):

    Samples=N_features
    #xx, yy = _som_class.get_euclidean_coordinates()
    xx, yy = build_lattice(_som_class)
    #weights_from_som = _som_class.get_weights()
    weights_from_som = _som_class.getWeights().reshape(xx.shape[0], xx.shape[1], Samples.shape[1])
    # target variable
    TL = target_labels
    markers = ['.', '.', '.','.','.','.']
    colors = ['C9', 'C0', 'C2','C8','C1','C3']

    
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')
    for i in range(weights_from_som.shape[0]):
        for j in range(weights_from_som.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius=.95 / np.sqrt(3),
                                 facecolor='white', 
                                 alpha=1, 
                                 edgecolor='gray')
            ax.add_patch(hex)

    # Plotting boundaries
    line_coordinates = find_boundaries_on_grid(_som_class=_som_class,
                            matplot_ax_grid_with_colored_dots=(f,ax),
                            hex_cluster_no_arr=nodes_cluster_no,
                            toroidal=False)
    for line in line_coordinates:
        (x1, y1), (x2, y2) = line
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=2.5)
        
    # Superimposing the samples
    Ws = samples_loc_on_grid 
    for cnt, x in enumerate(N_features):
        wx = Ws[cnt+1][0]
        wy = Ws[cnt+1][1]
        plt.plot(wx, wy, 
                 '.', 
                 markerfacecolor=colors[int(TL[cnt])],
                 markeredgecolor='None', 
                 markersize=8, 
                 markeredgewidth=2,
                 alpha=1)

    # Legend
    unique, counts = np.unique(TL, return_counts=True)
    in_d = dict(zip(unique, counts))
    legend_elements = [Line2D([0], [0], marker='.', color='C9', label=f'Excellent (n={in_d["0"]})',
                       markerfacecolor='C9', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C0', label=f'Very good (n={in_d["1"]})',
                       markerfacecolor='C0', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C2', label=f'Good (n={in_d["2"]})',
                       markerfacecolor='C2', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C8', label=f'Moderate (n={in_d["3"]})',
                       markerfacecolor='C8', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C1', label=f'Fair (n={in_d["4"]})',
                       markerfacecolor='C1', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C3', label=f'Poor (n={in_d["5"]})',
                       markerfacecolor='C3', markersize=14, linestyle='None', markeredgewidth=2)]
    ax.legend(title='Performance Level',
              handles=legend_elements, bbox_to_anchor=(1., 1.),
              loc='upper left', borderaxespad=0., ncol=1, fontsize=14)
    # Show plot
    #xrange = np.arange(weights.shape[0]); yrange = np.arange(weights.shape[1])
    #plt.xticks(xrange-.5, xrange); plt.yticks(yrange * np.sqrt(3) / 2, yrange)
    ax.axis('off')
    plt.show()


def le_kge(data_df):
    KGE_var = data_df['kge_categories'].apply(lambda x: '0' if x=='Excellent'
                                                   else ('1' if x=='Very good'
                                                         else ('2' if x=='Good'
                                                               else ('3' if x=='Moderate'
                                                                          else ('4' if x=='Fair'
                                                                                else '5')))))
    KGE_var
    y_label = np.asarray(KGE_var.apply(lambda x: int(x)))
    
    return y_label


def ST_clustering_workflow(data_df, hyperparam_dict, filename_tosave=''):

    # assume the data_df has not been normalized
    data_df = data_df.sort_values(by=['StaID'], ascending=True, inplace=False).reset_index(drop=True)
    TS_df = data_df[[str(i) for i in range(1,367)]]
    TS_df = (TS_df-TS_df.min())/(TS_df.max()-TS_df.min())
    TS_df = TS_df.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=1)
    
    _input_arr = TS_df.to_numpy()  # hard coded
    _Flow_Percentiles_df = TS_df   # hard coded

    xgird_size = hyperparam_dict['grid_x']
    ygrid_size = hyperparam_dict['grid_y']
    _window_size = hyperparam_dict['Wsize']
    no_of_clusters = hyperparam_dict['n_cluster']
    _niteration = hyperparam_dict['n_iteration']

    # Training the model 
    som_class = _somtimes_func(_input_arr, _window_size, xgird_size, ygrid_size, niteration=_niteration)
    _weights = som_class.getWeights().reshape(xgird_size, ygrid_size, 366)    # 366 is time series length
    samples_on_grid_locs =  getLocations(som_class, _input_arr, _window_size)
    
    #clear_output(wait=True)
    # Umatrix_plot(som_class, _Flow_Percentiles_df  ,_window_size)

    # Time series data clusters and color_coded grid
    grid_color_arr, nodes_cluster_number, samples_cluster_number = TScluster(_weights, _input_arr, samples_on_grid_locs, no_of_clusters)

    # color code the grid
    #sample_locs_on_grid_Ws = plot_colored_grid(som_class, _input_arr, grid_color_arr, nodes_cluster_number, samples_cluster_number, _window_size)

    
    csv_file_df = data_df[['StaID', 'lat', 'long', 'kge', 'kappa','nse','bal_accuracy']]
    csv_file_df['somtimes_cluster'] = samples_cluster_number
    # Save a csv file to the directory
    #file_name = 'somtimes_clusters'+'_'+str(_input_arr.shape[0])+'.csv'
    #if os.path.exists(file_name): # If it exists, delete the existing file
        #os.remove(file_name)
    #csv_file_df.to_csv(file_name)

    return csv_file_df, som_class

def TS_cluster_plot(samples_with_cluster_number):
    """
    This function receives the somtimes trained weights (2d numpy array),
    samples: time series data in the form of 2d array (each row is a time series)
    locations of samples on the som grid which is a list of tuples
    and returns a plot of time series cluster
    """

    #data = _st_weights
    n_cluster = samples_with_cluster_number['somtimes_cluster'].max()
    samples = samples_with_cluster_number[[str(i) for i in range(1,367)]].to_numpy()   # subjective
    # Clustering of weights (nodes) using h clustering
    # seed = 0
    #colors_of_nodes, cluster_no_of_nodes = poly_colors(data, number_of_clusters=n_cluster)
    #cluster_number_of_samples = get_samples_cluster_number(sample_locations, cluster_no_of_nodes)
    
    # Row and Column calculation based of number of clusters
    if n_cluster>20:
        print('Number of clusters could not be more than 7')
        return
    elif n_cluster%2 == 0:
        num_rows = int(n_cluster/2)
        num_cols = 2 
    else:
        num_rows = int(n_cluster/2) + 1
        num_cols = 2
    
    cluster_names_list = ['September-October Peak', 'June-July Decrease', 'June Peak with Gradual Decrease', 'Extended June-August Peak',
                          'March Peak with Decrease to Zero-flow', 'June Peak with Decrease to Zero-flow', 'April-May Peak', 'Complex Pattern']
    # Plotting clusters
    _figsize = (num_cols*4.5, num_rows*2.2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=_figsize, sharey=True, constrained_layout=True, frameon=False)
    #main_title = settitle(var_name)
    fig.suptitle("", fontsize=20,va='top',ha='center')
    clus_num = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if clus_num >= n_cluster:
                break
            
            samples_in_cluster_x_indices = np.where(np.asarray(samples_with_cluster_number['somtimes_cluster'])==clus_num+1)[0].tolist()
            for xx in samples[samples_in_cluster_x_indices, :]:
                axs[r, c].plot(xx, color = 'grey' , alpha=0.25)
            axs[r, c].plot(closest_to_median(samples[samples_in_cluster_x_indices, :]), color = _colors[clus_num])
            Nvalue = len(samples_in_cluster_x_indices)
            axs[r,c].set_title('Cluster '+ chr(clus_num+1+64) + ' (n='+str(Nvalue)+'),'+'\n'+ cluster_names_list[clus_num],fontsize=13.5)

            #axs[r,c].axvline(x=91, color='black', linestyle='--', linewidth=1)  # April (April starts at day 91 in a non-leap year)
            #axs[r,c].axvline(x=274, color='black', linestyle='--', linewidth=1)
            
            # monthly xticks
            axs[r,c].xaxis.set_major_locator(MonthLocator())
            axs[r,c].xaxis.set_major_formatter(DateFormatter('%b'))
            axs[r,c].set_xticks(axs[r,c].get_xticks()[:-1])
            #axs[r,c].grid(axis='x', linestyle='--', color='gray', alpha=0.5)
            

            clus_num += 1 
        
    if n_cluster%2!=0:
        axs[num_rows-1, num_cols-1].axis('off') 


#### Code starts here ####
# test some details (no need to run this since the output csv can be read from the directory as 'sometimes_cluster_376_14.csv'
_hyperparam_dict = {'grid_x': 10,
                    'grid_y': 10,
                    'Wsize': 15 ,
                    'n_cluster':14,
                    'n_iteration': 25}

_df, som_st_class = ST_clustering_workflow(data_df=Merged_df, hyperparam_dict=_hyperparam_dict)

# Merge the resulted csv file for sometimes cluster with the file that has the normalized streamflow drought signatures
_df = pd.read_csv(os.path.join(os.getcwd(), 'Data','somtimes_cluster_376_14.csv'))
drought_signatures_df = pd.read_csv(os.path.join(os.getcwd(), 'Data','normlaized_drought_signatures_376.csv'))

# combining clusters based on visual inspection as well as inter- and intra-cluster DTW distance of clusters
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({1:1, 6: 1}) 
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({2:2, 3: 2, 4: 2}) #
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({5: 3}) # 
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({7: 4}) # 
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({8: 5})
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({9:6, 10: 6, 11:6})
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({12: 7})
_df['somtimes_cluster'] = _df['somtimes_cluster'].replace({13: 8, 14: 8})

somtimes_df_merged_clusters_final = pd.merge(drought_signatures_df, _df[['StaID','somtimes_cluster']], on='StaID')

# plotting the final clusters
TS_cluster_plot(somtimes_df_merged_clusters_final)
