def nearest_neighbour(points):
    from numpy import meshgrid

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xg1, xg2 = meshgrid(xs, xs, indexing='ij')
    yg1, yg2 = meshgrid(ys, ys, indexing='ij')

    dx2, dy2 = (xg1 - xg2)**2, (yg1 - yg2)**2
    d = (dx2 + dy2)**0.5
    d[d==0] = 1e8

    return d.min(axis=1)

### Normalisations

def normalise(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def gaussian_normalisation(arr):
    from numpy import nanmean, nanstd
    return (arr - nanmean(arr)) / nanstd(arr)

def quantile_normalisation(arr):
    from numpy import quantile
    return (arr - quantile(arr, 0.5)) / (quantile(arr, 0.75) - quantile(arr, 0.25))

### Image Corrections

def ZtoRGB(channel):
    from numpy import uint8, where, isfinite, maximum, minimum
    y = maximum(minimum(channel, 5), -5)
    return uint8(where(isfinite(channel), 1 + (255-1) * (y - (-5)) / (5 - (-5)), 0))

from skimage.filters import gaussian
def segmental_correction(image, masks, flux_normalisation=gaussian_normalisation, blur=gaussian):
    from liams_funcs import decompose_masks
    from numpy import stack, nan, zeros_like, uint8

    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    R1, G1 ,B1 = zeros_like(R), zeros_like(G), zeros_like(B)
    for mask in decompose_masks(masks, reduce=False):
        R1[mask] = blur(ZtoRGB(flux_normalisation(R[mask])), preserve_range=True)
        G1[mask] = blur(ZtoRGB(flux_normalisation(G[mask])), preserve_range=True)
        B1[mask] = blur(ZtoRGB(flux_normalisation(B[mask])), preserve_range=True)

    return stack([R1, G1, B1], axis=2).astype(uint8)

def erodeNtimes(mask, N=1):
    from skimage.morphology import binary_erosion
    if N == 1: return binary_erosion(mask)
    return erodeNtimes(binary_erosion(mask), N=N-1)

### Classes

class Segment:
    def __init__(self, cell_rp, nucleus_rp, segment_order:int=0):
        from numpy import where, nan, quantile, clip, exp, pi
        from mahotas.features import haralick
        from pathtest import main_pathAnalysis
        from Functions_George import detect_outline

        self.cell_rp = cell_rp
        self.nucleus_rp = nucleus_rp
        self.segment_order = segment_order

        # Centre and neighbour

        self.x0, self.y0, self.x1, self.y1 = self.nucleus_rp.bbox
        self.xc, self.yc = self.nucleus_rp.centroid_local[0] + self.x0, self.nucleus_rp.centroid_local[1] + self.y0
        self.centre = (self.xc, self.yc)
        self.nearest_neighbour = 0

        # Colour channels
        self.cellR = where(self.cell_rp.image, self.cell_rp.intensity_image[:,:,0], nan)
        self.cellB = where(self.cell_rp.image, self.cell_rp.intensity_image[:,:,2], nan)
        self.nucleusR = where(self.nucleus_rp.image, self.nucleus_rp.intensity_image[:,:,0], nan)
        self.nucleusB = where(self.nucleus_rp.image, self.nucleus_rp.intensity_image[:,:,2], nan)

        # Binary channels
        self.binary_q = lambda q: clip((self.nucleusR > quantile(self.nucleusR[self.nucleus_rp.image], q)).astype(int) + detect_outline(self.nucleus_rp.image), 0, 1)
        self.binary_q50 = self.binary_q(0.5)

        # Haralick features

        self.haralickR = haralick(self.nucleus_rp.intensity_image[:,:,0], ignore_zeros=True).mean(0)
        self.haralickB = haralick(self.nucleus_rp.intensity_image[:,:,2], ignore_zeros=True).mean(0)

        # Possible paths

        #self.paths = main_pathAnalysis(self.binary_q50)

        # Functions

        self.quantile = quantile
        self.gaussian = lambda x, mu=128, sigma=25.4: exp(-(x - mu)**2 / (2 * sigma**2)) / (2 * pi * sigma**2)**0.5
        

    def show_segments(self, vmin=0, vmax=255):
        from matplotlib.pyplot import subplots, tight_layout, show, colorbar

        fig, axes = subplots(2, 2, figsize=(12,8), dpi=300)
        
        im = axes[0,0].imshow(self.cellR, origin='lower', extent=self.cell_rp.bbox, 
                              cmap='Reds', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[0,0], label='Cell redness', shrink=0.7)
        im = axes[0,1].imshow(self.cellB, origin='lower', extent=self.cell_rp.bbox, 
                              cmap='Blues', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[0,1], label='Cell blueness', shrink=0.7)
        im = axes[1,0].imshow(self.nucleusR, origin='lower', extent=self.nucleus_rp.bbox, 
                              cmap='Reds', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[1,0], label='Nucleus redness', shrink=0.7)
        im = axes[1,1].imshow(self.nucleusB, origin='lower', extent=self.nucleus_rp.bbox, 
                              cmap='Blues', vmin=vmin, vmax=vmax)
        colorbar(im, ax=axes[1,1], label='Nucleus blueness', shrink=0.7)

        tight_layout()
        show()

    ### Regionprops
        
    def getNucleusArea(self): return self.nucleus_rp.area
    def getCellArea(self): return self.cell_rp.area

    def getNucleusAreaBbox(self): return self.nucleus_rp.area_bbox
    def getCellAreaBbox(self): return self.cell_rp.area_bbox

    def getNucleusAreaConvex(self): return self.nucleus_rp.area_convex
    def getCellAreaConvex(self): return self.cell_rp.area_convex

    def getNucleusAxisMajor(self): return self.nucleus_rp.axis_major_length
    def getCellAxisMajor(self): return self.cell_rp.axis_major_length

    def getNucleusAxisMinor(self): return self.nucleus_rp.axis_minor_length
    def getCellAxisMinor(self): return self.cell_rp.axis_minor_length

    def getNucleusEcc(self): return self.nucleus_rp.eccentricity
    def getCellEcc(self): return self.cell_rp.eccentricity

    def getNucleusDiam(self): return self.nucleus_rp.equivalent_diameter_area
    def getCellDiam(self): return self.cell_rp.equivalent_diameter_area

    def getNucleusFeretDiam(self): return self.nucleus_rp.feret_diameter_max
    def getCellFeretDiam(self): return self.cell_rp.feret_diameter_max

    def getNucleusMaxR(self): return self.nucleusR[self.nucleus_rp.image].max()
    def getCellMaxR(self): return self.cellR[self.cell_rp.image].max()

    def getNucleusMinR(self): return self.nucleusR[self.nucleus_rp.image].min()
    def getCellMinR(self): return self.cellR[self.cell_rp.image].min()

    def getNucleusStdR(self): return self.nucleusR[self.nucleus_rp.image].std()
    def getCellStdR(self): return self.cellR[self.cell_rp.image].std()

    def getNucleusSolidity(self): return self.nucleus_rp.solidity
    def getCellSolidity(self): return self.cell_rp.solidity

    def getNucleusOri(self): return self.nucleus_rp.orientation
    def getCellOri(self): return self.cell_rp.orientation

    def getNucleusPerim(self): return self.nucleus_rp.perimeter
    def getCellPerim(self): return self.cell_rp.perimeter

    ### Haralick

    def getH1R(self): return self.haralickR[0]
    def getH2R(self): return self.haralickR[1]
    def getH3R(self): return self.haralickR[2]
    def getH4R(self): return self.haralickR[3]
    def getH5R(self): return self.haralickR[4]
    def getH6R(self): return self.haralickR[5]
    def getH7R(self): return self.haralickR[6]
    def getH8R(self): return self.haralickR[7]
    def getH9R(self): return self.haralickR[8]
    def getH10R(self): return self.haralickR[9]
    def getH11R(self): return self.haralickR[10]
    def getH12R(self): return self.haralickR[11]
    def getH13R(self): return self.haralickR[12]

    def getH1B(self): return self.haralickB[0]
    def getH2B(self): return self.haralickB[1]
    def getH3B(self): return self.haralickB[2]
    def getH4B(self): return self.haralickB[3]
    def getH5B(self): return self.haralickB[4]
    def getH6B(self): return self.haralickB[5]
    def getH7B(self): return self.haralickB[6]
    def getH8B(self): return self.haralickB[7]
    def getH9B(self): return self.haralickB[8]
    def getH10B(self): return self.haralickB[9]
    def getH11B(self): return self.haralickB[10]
    def getH12B(self): return self.haralickB[11]
    def getH13B(self): return self.haralickB[12]

    ### Custom 
    from math import pi

    def getNucleusRoundness(self, normalisation=4*pi): return normalisation * self.nucleus_rp.area / self.nucleus_rp.perimeter**2
    def getCellRoundness(self, normalisation=4*pi): return normalisation * self.cell_rp.area / self.cell_rp.perimeter**2

    def getNucleusFraction(self): return self.nucleus_rp.area / self.cell_rp.area

    def getSegmentOrder(self): return self.segment_order
    
    def getEnergyR(self): return (self.nucleusR[self.nucleus_rp.image]**2).sum()
    def getEnergyB(self): return (self.nucleusB[self.nucleus_rp.image]**2).sum()

    def getEdgeFluxR(self, N=5): return self.nucleusR[erodeNtimes(self.nucleus_rp.image, N=N)].mean()

    def getNObjects(self):
        from skimage.measure import label
        return label(self.binary_q50, return_num=True, connectivity=2)[1]
    
    def getEuler(self):
        from skimage.measure import euler_number
        return euler_number(self.binary_q50, connectivity=2)
    
    def getNHoles(self): return self.getNObjects() - self.getEuler()

    def getAreaOfHoles(self): return (self.binary_q50^self.nucleus_rp.image).astype(int).sum() / self.nucleus_rp.area

    def getEntropyR(self):
        from skimage.measure import shannon_entropy
        return shannon_entropy(self.nucleusR)
    
    def getEntropyB(self):
        from skimage.measure import shannon_entropy
        return shannon_entropy(self.nucleusB)
    
    def getLLH(self):
        from numpy import log, mean
        return -mean(log(self.gaussian(self.nucleusR[self.nucleus_rp.image])))
    
    def getNearestN(self): return self.nearest_neighbour / self.nucleus_rp.equivalent_diameter_area

    def getQ05(self): self.quantile(self.nucleusR[self.nucleus_rp.image], 0.05)

    def getQ25(self): self.quantile(self.nucleusR[self.nucleus_rp.image], 0.25)

    def getQ50(self): self.quantile(self.nucleusR[self.nucleus_rp.image], 0.50)

    def getQ75(self): self.quantile(self.nucleusR[self.nucleus_rp.image], 0.75)

    def getQ95(self): self.quantile(self.nucleusR[self.nucleus_rp.image], 0.95)


    #def getTotalPaths(self): return self.paths[0]
    #def getPathLength(self): return self.paths[1]
    
    ### Retrieving methods

    def retrieveFeatureNames(self): return [func for func in dir(self) if callable(getattr(self, func)) and func.startswith("get")]

    def retrieveFeatures(self): return [getattr(self, func)() for func in self.retrieveFeatureNames()]

class Masks:
    def __init__(self, fname, type='control', flux_normalisation=gaussian_normalisation):
        from cellpose.io import imread
        from numpy import load, arange, meshgrid
        from skimage.measure import regionprops

        self.type = type
        if self.type == 'control':
            image_dir = '../control_images/'
            mask_dir = '../segments/control/'
        else:
            image_dir = '../penetramax_images/'
            mask_dir = '../segments/penetramax/'

        # Load image and its masks
        self.image = imread(image_dir + fname)
        data = load(mask_dir + fname + '.npy')
        self.cells, self.nuclei = data[:,:,0], data[:,:,1]
        self.mask_vals = arange(1, self.cells.max()+0.5)

        # Correct images
        self.cell_images = segmental_correction(self.image, self.cells, flux_normalisation=flux_normalisation)
        self.nucleus_images = segmental_correction(self.image, self.nuclei, flux_normalisation=flux_normalisation)

        # Pixel coordinates
        self.xg, self.yg = meshgrid(arange(self.image.shape[0]), arange(self.image.shape[1]))

        # Genprops
        self.cell_regionprops = regionprops(self.cells, self.cell_images)
        self.nucleus_regionprops = regionprops(self.nuclei, self.nucleus_images)

        self.segments = []
        points = []
        for cell_rp, nucleus_rp in zip(self.cell_regionprops, self.nucleus_regionprops):
            seg = Segment(cell_rp, nucleus_rp)
            points.append(seg.centre)
            self.segments.append(seg)

        distances = nearest_neighbour(points)
        for seg, d in zip(self.segments, distances):
            seg.nearest_neighbour = d

    def getDataFrame(self):
        from pandas import DataFrame
        df = DataFrame()

        data = {}
        for label in self.segments[0].retrieveFeatureNames():
            data[label] = []

        if self.type == 'control':
            data['label'] = [0] * len(self.segments)
        else:
            data['label'] = [1] * len(self.segments)

        for mask in self.segments:
            for label, val in zip(mask.retrieveFeatureNames(), mask.retrieveFeatures()):
                data[label].append(val)

        df = df.from_dict(data)
        self.df = df
        return self.df

###
    
def inverse_quantile(vals):
    return vals.argsort() / vals.argsort().max()

from sklearn.preprocessing import StandardScaler
class Dataset:
    def __init__(self, control_paths, penetramax_paths, scaler=StandardScaler):
        from pandas import concat
        from tqdm import tqdm

        self.control_paths = control_paths
        self.penetramax_paths = penetramax_paths

        # Create list of Masks objects
        print("Instantiating masks...")
        self.masks = []
        loop = tqdm(zip(self.control_paths + self.penetramax_paths, len(self.control_paths) * ['control'] + len(self.penetramax_paths) * ['penetramax']))
        for fname, type in loop:
            self.masks.append(Masks(fname.split('/')[-1], type=type))

        # Combined feature dataframes from each object
        print("Retrieving features...")
        dfs = []
        for mask in tqdm(self.masks):
            dfs.append(mask.getDataFrame())
        self.df = concat(dfs)
        self.feature_names = self.masks[0].segments[0].retrieveFeatureNames()
        self.feature_df = self.df[self.feature_names]

        # X-matrices
        self.X = self.feature_df.to_numpy()
        self.scaler = scaler().fit(self.X)
        self.X_scaled = self.scaler.transform(self.X)
        self.y = self.df['label'].to_numpy()

        self.X_reduced = None
        self.X_control_reduced = None
        self.X_penetramax_reduced = None

    from sklearn.decomposition import PCA
    def performDimReduction(self, n_components:int=3, Algo=PCA):
        self.n_components = n_components
        algo = Algo(n_components=self.n_components)

        is_control = (self.y==0)
        X_control = self.X[is_control,:]
        X_penetramax = self.X[~is_control,:]

        self.X_control_reduced = algo.fit_transform(X_control)
        self.X_penetramax_reduced = algo.fit_transform(X_penetramax)
        self.X_reduced = algo.fit_transform(self.X)

    def makeKDE(self):
        from scipy.stats import gaussian_kde
        from numpy import append

        if self.X_control_reduced is None:
            print('Run dimensionality reduction first!..')
            return
        
        if self.X_control_reduced.shape[0] == self.n_components: flip = lambda x: x
        else: flip = lambda x: x.T

        kde_control = gaussian_kde(flip(self.X_control_reduced))
        kde_penetramax = gaussian_kde(flip(self.X_penetramax_reduced))

        control_LLHs = inverse_quantile(kde_control.logpdf(flip(self.X_control_reduced)))
        drug_LLHs = inverse_quantile(kde_penetramax.logpdf(flip(self.X_penetramax_reduced)))

        self.df['Similarity'] = append(control_LLHs, drug_LLHs)

    def makeSelection(self, q_control=0.05, q_drug=0.05):
        condition = (self.df['label'] == 0) * (self.df['Similarity'] >= q_control) + (self.df['label'] == 1) * (self.df['Similarity'] >= q_drug)
        df_reduced = self.df[condition]
        return df_reduced[self.feature_names + ['label']]