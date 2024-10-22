from minisom import MiniSom
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import RegularPolygon, Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D
import matplotlib as mpl
import decimal
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os
import json
import pickle
import scikit_posthocs as sp
from scipy import stats
from sklearn.preprocessing import normalize

watershed_attributes_50 = [ 
       'DA_SQKM', 'MAXDI_EROM', 'Dam_Index', 'TOT_ELEV_MEAN', 'TOT_ELEV_MAX',
       'TOT_STREAM_SLOPE', 'TOT_MAXP6190', 'TOT_MAXWD6190', 'TOT_MINWD6190',
       'TOT_RH', 'TOT_AET', 'TOT_CWD', 'TOT_BFI', 'TOT_CONTACT', 'TOT_IEOF',
       'TOT_RECHG', 'TOT_SATOF', 'TOT_TWI', 'TOT_EWT', 'TOT_RF7100',
       'TOT_MIRAD_2012', 'TOT_FRESHWATER_WD', 'TOT_STREAMRIVER',
       'TOT_ARTIFICIAL', 'TOT_CONNECTOR', 'TOT_STRM_DENS',
       'TOT_TOTAL_ROAD_DENS', 'TOT_HGA', 'TOT_HGB', 'TOT_HGC', 'TOT_HGD',
       'TOT_SILTAVE', 'TOT_CLAYAVE', 'TOT_SANDAVE', 'TOT_KFACT',
       'TOT_KFACT_UP', 'TOT_NO10AVE', 'TOT_NO200AVE', 'TOT_OM', 'TOT_ROCKDEP',
       'TOT_BDAVE', 'TOT_WTDEP', 'TOT_SRL25AG', 'TOT_NLCD19_31',
       'TOT_NLCD19_41', 'TOT_NLCD19_43', 'TOT_NLCD19_71', 'TOT_NLCD19_81',
       'TOT_NLCD19_FOREST', 'TOT_NLCD19_WETLAND']

List_of_wa_names = ['Drainage area', 'Reservoir storage intensity', 'Degree of regulation', 'Mean elevation', 'Max elevation',
                    'Stream slope', 'Max precipitation','Max wet days', 'Min wet days', 'Relative humadity', 'Evapotraspiration',
                    'Consecutive wet days', 'Baseflow index', 'Contact time index', 'Horton flow', 'Ground water recharge',
                    'Dunne flow', 'Topgraphic wetness index', 'Water table depth', 'Rainfall-runoff factor', 'Irrigated agriculture',
                    'Freshwater withdrawal', 'Percent reach length', 'Percent artificial reach', 'Percent connector reach', 
                    'Stream density', 'Raod density', 'Hydrologic group A soil',  'Hydrologic group B soil',  'Hydrologic group C soil',
                    'Hydrologic group D soil', 'Percent silt', 'Percent clay', 'Percent sand', 'K factor', 'Upper soil K',
                    'Percent grain size <2 mm', 'Percent grain size <0.74 mm', 'Organic matter', 'Soil thickness', 'Bulk density',
                    'High water table depth', 'Soil restrective layer', 'Percent barren land', 'Percent deciduous forest',
                    'Percent mixed forest', 'Percent grassland cover', 'Percent pasture/hay', 'Percent forest', 'Percent wetland']

node_colors = ['#1619f0', '#ba27cd', '#a3201d', '#0df4f0', '#33a02c', '#ff7f00', '#f6f918', 'darkorange','olivedrab', 'blue','red', 'black']
global node_colors
global watershed_attributes_50


### Functions ###
# functions
def train_som(data_array, epochs, nrows, ncols, inp_len, sigma, lr):
    
    som = MiniSom(x=nrows,
              y=ncols,
              input_len= inp_len,
              sigma=sigma,
              learning_rate=lr,
              activation_distance='euclidean',
              topology='hexagonal',
              neighborhood_function='gaussian',
              random_seed=10)
    
    som.pca_weights_init(data_array)
    st_time = time.time()
    som.train_random(data_array, epochs)
    elapsed_time = time.time() - st_time
    print(elapsed_time,'seconds')
    
    return som


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


def round_up(x, place=0):
    
    context = decimal.getcontext()
    # get the original setting so we can put it back when we're done
    original_rounding = context.rounding
    # change context to act like ceil()
    context.rounding = decimal.ROUND_CEILING
    rounded = round(decimal.Decimal(str(x)), place)
    context.rounding = original_rounding
    
    return float(rounded)


def poly_colors(_som_weights, number_of_clusters):
    
    _som_grid_x = _som_weights.shape[0] # ncols
    _som_grid_y = _som_weights.shape[1] # nrows
    _som_grid_z = _som_weights.shape[2] # n features
    _som_n_nodes = int(_som_grid_x*_som_grid_y)
    
    # H clustering of weights
    linkage_matrix = linkage(_som_weights.reshape(_som_n_nodes,_som_grid_z), method='ward', metric='euclidean')
    cluster_assignments = fcluster(linkage_matrix, t=number_of_clusters, criterion='maxclust')
    
    # creating a corresponding color list with the cluster assignment
    _colors = node_colors
    color_labels = [_colors[val-1] for val in cluster_assignments]
    
    return np.asarray(color_labels).reshape(_som_grid_x, _som_grid_y), cluster_assignments


def get_samples_cluster_number(grid_size, raw_samples_on_grid_loc, hcluster_assignments):
    
    samples_cluster_number = []
    for loc in raw_samples_on_grid_loc:
        samples_cluster_number.append(hcluster_assignments[np.ravel_multi_index(loc, dims=grid_size)]) # dims hardcoded
        
    return samples_cluster_number


def plot_colored_grid(_som_class, N_features, no_clusters):

    xx, yy = _som_class.get_euclidean_coordinates()
    weights_from_som = _som_class.get_weights()
    pgcolors, hclus_assignments = poly_colors(_som_weights=weights_from_som, number_of_clusters=no_clusters)
    
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
        # getting the winner
        w = _som_class.winner(x)
        samples_on_grid_loc.append(w)
        # place a marker on the winning position for the sample xx
        wx, wy = _som_class.convert_map_to_euclidean(w) 
        wy = wy * np.sqrt(3) / 2
        wx, wy, Ws = loc_check(wx, wy, Ws, [-0.15, 0.15])
    
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
    plt.savefig('SOMGrid_color_coded.png', dpi=300, transparent=True)
    plt.show()

    return hclus_assignments, get_samples_cluster_number(weights_from_som.shape[0:2], samples_on_grid_loc, hcluster_assignments=hclus_assignments), Ws


def grid_with_colored_dots(_som_class, N_features, samples_loc_on_grid, nodes_cluster_no, target_labels):

    xx, yy = _som_class.get_euclidean_coordinates()
    weights_from_som = _som_class.get_weights()
    
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
            if (i,j)==(0,0):
                ax.plot(xx[(i, j)], wy,'o','red')

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
                 markers[TL[cnt]], 
                 markerfacecolor=colors[TL[cnt]],
                 markeredgecolor='None', 
                 markersize=12, 
                 markeredgewidth=2,
                 alpha=1)

    # Legend
    unique, counts = np.unique(TL, return_counts=True)
    in_d = dict(zip(unique, counts))
    legend_elements = [Line2D([0], [0], marker='.', color='C9', label=f'Very good to excellent (n={in_d[0]})',
                       markerfacecolor='C9', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C0', label=f'Moderate to good (n={in_d[1]})',
                       markerfacecolor='C0', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C2', label=f'Good (n={in_d[2]})',
                       markerfacecolor='C2', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C8', label=f'Moderate (n={in_d[3]})',
                       markerfacecolor='C8', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C1', label=f'Fair (n={in_d[4]})',
                       markerfacecolor='C1', markersize=14, linestyle='None', markeredgewidth=2),
                       Line2D([0], [0], marker='.', color='C3', label=f'Poor (n={in_d[5]})',
                       markerfacecolor='C3', markersize=14, linestyle='None', markeredgewidth=2)]
    ax.legend(title='Performance Level',
              handles=legend_elements, bbox_to_anchor=(1., 0.9),
              loc='upper left', borderaxespad=0., ncol=1, fontsize=14)
    # Show plot
    #xrange = np.arange(weights.shape[0]); yrange = np.arange(weights.shape[1])
    #plt.xticks(xrange-.5, xrange); plt.yticks(yrange * np.sqrt(3) / 2, yrange)
    ax.axis('off')
    plt.show()
  

def som_clustering_pipeline(data_df, list_of_WA, hyperparams_dict):

    # normalizing data_df columns (watershed attributes)
    data_df = data_df.sort_values(by=['StaID'], ascending=True, inplace=False).reset_index(drop=True)
    WA_df = data_df[list_of_WA]
    WA_df=(WA_df-WA_df.min())/(WA_df.max()-WA_df.min()) # normalizing watershed attributes
    WA_arr = WA_df.to_numpy()
    print(f' sample size: {WA_arr.shape}') # to make sure data is in the correct format
    
    # SOM hyperparameters
    sigma=hyperparams_dict['sigma']
    lr=hyperparams_dict['lr']
    xdim = hyperparams_dict['xdim'] # nrows
    ydim = hyperparams_dict['ydim'] # ncols
    epochs = hyperparams_dict['epochs']
    inp_len = WA_arr.shape[1]

    # Training
    print(f'Training the SOM with a {ydim}*{xdim} lattice')
    _som_class = train_som(WA_arr, epochs, xdim, ydim, inp_len, sigma, lr)
    _weights = _som_class.get_weights()
    print(_weights.shape)

    # Clustering nodes and coloring them based on cluster number
    #ncluster = optimal_no_clusters(_weights) # between 2 to 10 clusters
    ncluster = hyperparams_dict['h_cluster_no']
    nodes_cluster_no, samples_cluster_no, samples_loc_on_grid = plot_colored_grid(_som_class, WA_arr, ncluster)

    # Plot grid with color_coded dots and boundaries
    #_TL = le_kge(data_df)
    #grid_with_colored_dots(_som_class, WA_arr, samples_loc_on_grid, nodes_cluster_no, target_labels=_TL)

    # Save a csv file to the directory
    csv_file_df = data_df[['StaID', 'lat', 'long'] + list_of_WA + ['nse', 'kge', 'kge_categories', 'kappa']]
    csv_file_df['som_cluster'] = samples_cluster_no
    file_name = 'WA_with_som_cluster'+'_'+str(WA_arr.shape[0])+'_'+str(len(list_of_WA))+'_1'+'.csv'   # output csv filename with som clusters
    if os.path.exists(file_name): # If it exists, delete the existing file
        os.remove(file_name)
    csv_file_df.to_csv(file_name)
    print(f'file saved to the {file_name} directory')

    return csv_file_df, _som_class


def loc_check(wxx, wyy, w_lst=None, x_rand=[-0.15, 0.15]):
    
    if w_lst==[]:
        w_lst=[(100, 100)]
    
    init_wxx = wxx
    init_wyy = wyy
    
    while [1 for i in w_lst if i==(wxx,wyy)]:
        px = np.random.choice(x_rand)
        py = np.random.choice(x_rand)
        if  (init_wxx+0.4 > wxx + px) & (init_wyy+0.4 > wyy + py):
            wxx = wxx + px
            wyy = wyy + py
        
    w_lst.append((wxx, wyy))
    
    return wxx, wyy, w_lst


def Umatrix_plot(_som_class, Samples):

    #V_max = round_up(np.max(weights_from_som),1)
    # Normalizing weights (comment this to have between zero and 1 reflectance values in hexagonals)
    #w = (w-np.min(w))/(V_max-np.min(w))

    xx, yy = _som_class.get_euclidean_coordinates()
    umatrix = _som_class.distance_map(scaling='mean')
    weights = _som_class.get_weights()

    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    ax.set_aspect('equal')

    # iteratively add hexagons
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            wy = yy[(i, j)] * np.sqrt(3) / 2
            hex = RegularPolygon((xx[(i, j)], wy), 
                                 numVertices=6, 
                                 radius=.95 / np.sqrt(3),
                                 facecolor=cm.Greys(umatrix[i, j]), 
                                 alpha=1, 
                                 edgecolor='gray')
            ax.add_patch(hex)
    Ws= []
    for cnt, x in enumerate(Samples):
        # getting the winner
        w = _som_class.winner(x)
        # place a marker on the winning position for the sample xx
        wx, wy = _som_class.convert_map_to_euclidean(w) 
        wy = wy * np.sqrt(3) / 2
        wx, wy, Ws = loc_check(wx, wy, w_lst=Ws)
    
    #wx = Ws[cnt][0]
    #wy = Ws[cnt][1]
    
        plt.plot(wx, wy, 
                 '.', 
                 markerfacecolor='None',
                 markeredgecolor='None', 
                 markersize=8, 
                 markeredgewidth=2,
                 alpha=0.5)

    ax.axis('off')
    plt.show()


#### Code starts here ####

# Importing spatial info
all_data = pd.read_csv(os.path.join(os.getcwd(), 'Data', 'KGE_VR_with_Categories.csv')
WA_df = all_data[watershed_attributes_50]
WA_df=(WA_df-WA_df.min())/(WA_df.max()-WA_df.min()) # normalizing watershed attributes
WA_arr = WA_df.to_numpy()


# all 384 gages with 50 features
HP_dict = {
    'sigma': 5,
    'lr': 0.7,
    'xdim': 12,
    'ydim':8,
    'epochs':2000,
    'h_cluster_no':7}

df, som_384_50 = som_clustering_pipeline(data_df = all_data, list_of_WA=watershed_attributes_50, hyperparams_dict=HP_dict)
