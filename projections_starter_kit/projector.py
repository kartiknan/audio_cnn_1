
from fastai.vision import *
from sklearn.decomposition import PCA
from tqdm import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from .actgetter import *

def getImage(img, zoom, alpha):
    return OffsetImage(image2np(img.data),zoom=zoom, alpha=alpha)

def class_colours(number_classes, cm_type='nipy_spectral'):
    cm = plt.get_cmap(cm_type)
    # random color selection from colour map
    cm_subsection = np.linspace(0, 255, number_classes)
    colours = [cm(int(x)) for x in cm_subsection][:number_classes]
    return [colours[i] for i in range(number_classes)]
    
def im_scatter(proj, x, y, targets, zooms, alphas, imgs, ax, show_images, show_legend, permute=True):
    '''
    Adds images from imgs at the give x, y coordinated to ax.
    '''
    
    if show_images:
        ax.scatter(x, y, s=1, c='white')  
        # Optional permultation, so that images receive different alpha
        # and zoom values at different plots
        artists = []; i = 0
        n = len(imgs)
        perm = np.random.permutation(n) if permute else np.arange(n)
        zipped = np.array(list(zip(x, y, zooms[perm], alphas[perm], imgs)))
        
        for x0, y0, z0, a0, img in zipped: 
            ab = AnnotationBbox(getImage(img, z0, a0), (x0, y0), frameon=False)
            artists.append(ax.add_artist(ab))
    else:
        number_classes = len(proj.dl.classes)
        cc = class_colours(number_classes, cm_type='nipy_spectral')
        for i in range(number_classes):
            class_indexs = (targets == i).nonzero()
            label = proj.dl.classes[i]
            plot_x, plot_y = x[class_indexs], y[class_indexs]
            ax.scatter(plot_x, plot_y, s=20, c=[cc[i]], label=label)
        
        if show_legend:
            ax.legend(prop={'size': 13})
            plt.show()
        
        
def create_zooms_alphas(n):
    
    '''Returns an array for zooms and alphas for our images'''
        
    zooms,  alphas = [], []
    
    for i in range(n):
        if i <= 50:  # First 50 images are larger, 100% alpha
            zooms.append(np.random.rand()*0.2 + 0.2)
            alphas.append(1)
        elif i <= 150:  # Next 100 are smaller, 70-100% alpha
            zooms.append(np.random.rand()*0.2 + 0.05)
            alphas.append(np.random.rand()*0.3 + 0.7)
        else:  # The rest is very small
            zooms.append(np.random.rand()*0.15 + 0.02)
            alphas.append(np.random.rand()*0.4 + 0.6)
    return np.array(zooms), np.array(alphas)        
        
class Projector():
    
    '''
    This class creates projections similar to platform.ai. 
    
    It takes the image activations, reduces them (using either UMAP or PCA), and plots
    two at a time.
    '''
    
    def __init__(self, model, target_layers, dl=None, **kwargs):
        self.eps=1e-5
        self.imgs, self.img_acts, self.red_method_str, self.targets = None, None, None, None
        self.act_getter = ActGetter(model, target_layers, **kwargs)
        if dl is not None: 
            self.proc_imgs(dl)
            self.dl = dl 
        
    def proc_imgs(self, dl):        
        self.imgs, self.img_acts = [], []
        self.targets = []
        for xb, yb in tqdm(dl):
            self.img_acts.append(self.act_getter(xb))
            self.targets.append(yb)
            for row in range(xb.shape[0]):  # batch size
                self.imgs.append(Image(xb[row].cpu()))
        self.imgs = np.array(self.imgs)
        self.img_acts = torch.cat(self.img_acts, dim=0)        
        self.targets = torch.cat(self.targets, dim=0)

        self.act_red = self.reduce_acts(self.img_acts)

    def refresh_activations(self):
        self.act_getter.clear_all_activations()
        self.proc_imgs(self.dl)
    
    def reduce_acts(self, tensor_activations, dim_red=10, red_method=PCA, init_red_method=True):
        '''Reduces image activations to dim_red dimensions'''
        self.dim_red, self.counter, self.red_method_str = dim_red, 0, str(red_method)

        # ensure list input 
        if type(dim_red) != list and type(red_method) != list:
            dim_red = [dim_red]
            red_method = [red_method]
        
        activations = tensor_activations.cpu().numpy()
        for reduction_method, dim_reduction in zip(red_method, dim_red):
            # init methods if not already 
            if init_red_method: reduction_method = reduction_method(n_components=dim_reduction)
            # normalize
            act_norm = (activations - activations.mean(0))/(self.eps+activations.std(0))
            # apply dimensionality reduction
            activations = reduction_method.fit_transform(act_norm)
        
        return activations

    def reset_counter(self):  self.counter=0
        
    def plot(self, axis = (0, 1), activations=None, dim_red=20, red_method=PCA, init_red_method=True, re_project=True, **kwargs):
        
        '''
        Plots two dimensions of a projection.
        
        Inputs:
        - axis: a tuple giving which dimensions we plot (from the reduced representation).
        - red: UMAP or PCA
        - savefig: The filepath where the figure will be saved. If None, figure isn't saved.
        '''

        if re_project:
            activations = self.img_acts if type(activations) == type(None) else activations
            act_red = self.reduce_acts(activations, dim_red, red_method, init_red_method)
            self.act_red = act_red
        else:
            act_red = self.act_red

        plot_x, plot_y = act_red[:, axis[0]], act_red[:, axis[1]]
        
        return self.create_fig_ax(plot_x, plot_y, **kwargs)

        
    def plot_next(self, **kwargs):
        
        '''Used to cycle through projections. Calling plot_next plots the next 2 dimensions.'''
        
        out = self.plot(axis = (self.counter, self.counter+1), **kwargs)
        
        self.counter = (self.counter + 2) % (self.act_red.shape[-1] - 1)
        
        return out
        
        
    def create_fig_ax(self, plot_x, plot_y, permute=True,
                      zoom_x = None, zoom_y = None,
                      title=None, xlabel=None, ylabel=None, 
                      figsize=(12, 12), savefig=None, 
                      return_fig_ax=False, show_images=True,
                      show_legend=True):
        
        fig, ax = plt.subplots(1, figsize=figsize)
        
        subs = np.array([True]*len(self.imgs))
        if zoom_x is not None: subs[(plot_x < zoom_x[0]) | (plot_x > zoom_x[1])] = False
        if zoom_y is not None: subs[(plot_y < zoom_y[0]) | (plot_y > zoom_y[1])] = False    
            
        zooms, alphas = create_zooms_alphas(subs.sum())  
        
        im_scatter(self, plot_x[subs], plot_y[subs], self.targets.cpu().numpy()[subs], zooms, alphas, self.imgs[subs], ax, show_images, show_legend, permute=permute)
        
        if title is not None: ax.set_title(title)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if xlabel is not None: 
            ax.spines['bottom'].set_visible(True)
            ax.set_xlabel(xlabel)
        else:
            ax.spines['bottom'].set_visible(False)
            ax.get_xaxis().set_ticks([])
             
        if ylabel is not None: 
            ax.spines['left'].set_visible(True)
            ax.set_ylabel(ylabel)
        else: 
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_ticks([])
        
        if savefig is not None:
            fig.savefig(savefig)
        if return_fig_ax:
            return fig, ax
        else: 
            return None