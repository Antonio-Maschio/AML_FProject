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

def make_square(image):
    from numpy import max, pad, uint8
    
    # Determine the original width and height
    h0, w0 = image.shape[:2]
    if h0 == w0:
        return image
    
    # Calculate padding for width and height
    pad_top = max([(w0 - h0) // 2, 0])
    pad_bottom = max([(w0 - h0) - pad_top, 0])
    pad_left = max([(h0 - w0) // 2, 0])
    pad_right = max([(h0 - w0) - pad_left, 0])
    
    # Apply padding and return
    return uint8(pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0))

### Image Corrections

def ZtoRGB(channel):
    from numpy import uint8, where, isfinite, maximum, minimum
    y = maximum(minimum(channel, 5), -5)
    return uint8(where(isfinite(channel), 1 + (255-1) * (y - (-5)) / (5 - (-5)), 0))

from skimage.filters import gaussian
def segmental_correction(image, masks, flux_normalisation=gaussian_normalisation, blur=gaussian):
    from liams_funcs import decompose_masks
    from numpy import stack, zeros_like, uint8

    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    R1, G1 ,B1 = zeros_like(R), zeros_like(G), zeros_like(B)

    if flux_normalisation is not None:
        transformation = lambda x: ZtoRGB(flux_normalisation(x))
    else:
        transformation = lambda x: x

    for mask in decompose_masks(masks, reduce=False):
        R1[mask] = blur(transformation(R[mask]), preserve_range=True)
        G1[mask] = blur(transformation(G[mask]), preserve_range=True)
        B1[mask] = blur(transformation(B[mask]), preserve_range=True)

    return stack([R1, G1, B1], axis=2).astype(uint8)

def erodeNtimes(mask, N=1):
    from skimage.morphology import binary_erosion
    if N == 1: return binary_erosion(mask)
    return erodeNtimes(binary_erosion(mask), N=N-1)

### Custom PyTorch Dataset

def flip_and_rotate(image, flip:bool=True, angles=[0, 90, 180, 270]):
    from skimage.transform import rotate
    from numpy import uint8

    if flip:
        flipped = image[::-1,:].copy()
        flipped_and_rotated = flip_and_rotate(flipped, flip=False, angles=angles)
    else:
        flipped_and_rotated = []

    rotated_images = []
    for angle in angles:
        rotated_images.append(uint8(255 * rotate(image, angle)))

    return rotated_images + flipped_and_rotated

from torch.utils.data import Dataset
class ImageDataset(Dataset):
    def __init__(self, image_type='nucleus'):
        from os import listdir
        from torchvision.io import read_image

        if image_type == 'nucleus':
            self.data_path = 'pytorch_data/nucleus/'
        elif image_type == 'cell':
            self.data_path = 'pytorch_data/cell/'

        self.control_dir = self.data_path + 'control/'
        self.drug_dir = self.data_path + 'drug/'

        self.images = []
        self.labels = []
        for fname in listdir(self.control_dir):
            self.images.append(read_image(self.control_dir + fname))
            self.labels.append(0)
        for fname in listdir(self.drug_dir):
            self.images.append(read_image(self.drug_dir + fname))
            self.labels.append(1)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

### Classes

class Segment:
    def __init__(self, cell_rp, nucleus_rp, segment_order:int=0):
        from numpy import where, nan, clip, exp, pi, quantile, stack, zeros_like, isnan, uint8
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

        cellRGB = stack([self.cellR, self.cellB, self.cellB], axis=2)
        nucleusRGB = stack([self.nucleusR, self.nucleusB, self.nucleusB], axis=2)

        cellRGB[isnan(cellRGB)] = 0
        nucleusRGB[isnan(nucleusRGB)] = 0

        self.cellRGB = uint8(cellRGB)
        self.nucleusRGB = uint8(nucleusRGB)

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

    def squareImage(self, final_size:int=0):
        from numpy import uint8
        from skimage.transform import resize

        im1 = self.cellRGB
        im2 = self.nucleusRGB

        # Make square
        im1_sq = make_square(im1)
        im2_sq = make_square(im2)

        # Resize
        if final_size <= 0:
            return im1_sq, im2_sq
        return uint8(255 * resize(im1_sq, (final_size, final_size))), uint8(255 * resize(im2_sq, (final_size, final_size)))

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

    def getQ05(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.05)

    def getQ25(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.25)

    def getQ50(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.50)

    def getQ75(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.75)

    def getQ95(self): return self.quantile(self.nucleusR[self.nucleus_rp.image], 0.95)

    def getSymR(self): return abs(self.quantile(self.nucleusR[self.nucleus_rp.image], 0.5) - self.nucleusR[self.nucleus_rp.image].mean())

    def getSymB(self): return abs(self.quantile(self.nucleusB[self.nucleus_rp.image], 0.5) - self.nucleusB[self.nucleus_rp.image].mean())

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
        from liams_funcs import get_mask_levels, get_generations

        self.type = type
        if self.type == 'control':
            image_dir = '../control_images/'
            mask_dir = 'segments/control/'
        else:
            image_dir = '../penetramax_images/'
            mask_dir = 'segments/penetramax/'

        # Load image and its masks
        self.image = imread(image_dir + fname)
        data = load(mask_dir + fname + '.npy')
        self.cells, self.nuclei = data[:,:,0], data[:,:,1]
        self.levels = get_mask_levels(self.nuclei)
        self.generations = get_generations(self.levels)

        # Correct images
        self.cell_images = segmental_correction(self.image, self.cells, flux_normalisation=flux_normalisation)
        self.nucleus_images = segmental_correction(self.image, self.nuclei, flux_normalisation=flux_normalisation)

        # Genprops
        self.cell_regionprops = regionprops(self.cells, self.cell_images)
        self.nucleus_regionprops = regionprops(self.nuclei, self.nucleus_images)

        self.segments = []
        points = []
        for cell_rp, nucleus_rp, generation in zip(self.cell_regionprops, self.nucleus_regionprops, self.generations):
            seg = Segment(cell_rp, nucleus_rp, segment_order=generation)
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

        images_R = []; images_B = []
        for mask in self.segments:
            images_R.append(mask.nucleusR)
            images_B.append(mask.nucleusB)
            for label, val in zip(mask.retrieveFeatureNames(), mask.retrieveFeatures()):
                data[label].append(val)

        df = df.from_dict(data)
        self.df = df
        self.images_R = images_R
        self.images_B = images_B
        return self.df

    def getImages(self, final_size:int=-1):
        cell_images = [mask.squareImage(final_size=final_size)[0] for mask in self.segments]
        nucleus_images = [mask.squareImage(final_size=final_size)[1] for mask in self.segments]
        if self.type == 'control':
            labels = [0] * len(cell_images)
        else:
            labels = [1] * len(cell_images)
        return (cell_images, nucleus_images), labels
    
###
    
def inverse_quantile(arr):
    from scipy.stats import percentileofscore
    return percentileofscore(arr, arr)

from sklearn.preprocessing import StandardScaler
class Dataset:
    def __init__(self, control_paths, penetramax_paths, scaler=StandardScaler, flux_normalisation=gaussian_normalisation):
        from pandas import concat
        from tqdm import tqdm

        self.control_paths = control_paths
        self.penetramax_paths = penetramax_paths

        # Create list of Masks objects
        print("Instantiating masks...")
        self.masks = []
        loop = tqdm(zip(self.control_paths + self.penetramax_paths, len(self.control_paths) * ['control'] + len(self.penetramax_paths) * ['penetramax']))
        for fname, type in loop:
            self.masks.append(Masks(fname.split('/')[-1], type=type, flux_normalisation=flux_normalisation))

        # Combined feature dataframes from each object
        print("Retrieving features...")
        dfs = []
        self.images_R = []
        self.images_B = []
        for mask in tqdm(self.masks):
            df = mask.getDataFrame()
            dfs.append(df)
            self.images_R += mask.images_R
            self.images_B += mask.images_B

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

        self.is_control = (self.y==0)

        self.control_images_R = [self.images_R[i] for i, b in enumerate(self.is_control) if b]
        self.control_images_B = [self.images_B[i] for (i, b) in enumerate(self.is_control) if b]
        self.drug_images_R = [self.images_R[i] for i, b in enumerate(self.is_control) if not b]
        self.drug_images_B = [self.images_B[i] for (i, b) in enumerate(self.is_control) if not b]

    from sklearn.decomposition import PCA
    def performDimReduction(self, n_components:int=2, Algo=PCA):
        self.n_components = n_components
        algo = Algo(n_components=self.n_components)

        self.X_control = self.X_scaled[self.is_control,:]
        self.X_penetramax = self.X_scaled[~self.is_control,:]

        self.X_control_reduced = algo.fit_transform(self.X_control)
        self.X_penetramax_reduced = algo.fit_transform(self.X_penetramax)
        self.X_reduced = algo.fit_transform(self.X_scaled)

    def makeKDE(self, show_plot:bool=False, save_to=None, resolution:int=100, threshold=0):
        from scipy.stats import gaussian_kde
        from numpy import append, meshgrid, linspace, stack, quantile
        from matplotlib.pyplot import subplots, tight_layout, show, savefig, colorbar
        from matplotlib.colors import ListedColormap, BoundaryNorm

        if self.X_control_reduced is None:
            print('Run dimensionality reduction first!..')
            return
        
        if self.X_control_reduced.shape[0] == self.n_components: flip = lambda x: x
        else: flip = lambda x: x.T

        kde_control = gaussian_kde(flip(self.X_control_reduced))
        kde_penetramax = gaussian_kde(flip(self.X_penetramax_reduced))
        kde_control_combined = gaussian_kde(flip(self.X_reduced[self.is_control,:]))
        kde_drug_combined = gaussian_kde(flip(self.X_reduced[~self.is_control,:]))

        control_LLHs = kde_control.logpdf(flip(self.X_control_reduced))
        drug_LLHs = kde_penetramax.logpdf(flip(self.X_penetramax_reduced))

        self.control_qLLHs = inverse_quantile(control_LLHs)
        self.drug_qLLHs = inverse_quantile(drug_LLHs)

        self.df['Similarity'] = append(self.control_qLLHs, self.drug_qLLHs)

        if show_plot:
            xs, ys = linspace(-15, 15, resolution, endpoint=True), linspace(-15, 15, resolution, endpoint=True)
            xg, yg = meshgrid(xs, ys, indexing='ij')

            zg1 = kde_control.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution)
            zg2 = kde_penetramax.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution)
            zg3 = kde_control_combined.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution) - kde_drug_combined.logpdf(stack([xg.flatten(), yg.flatten()], axis=0)).reshape(resolution, resolution)

            qs = [0, 0.05, 0.1, 0.25, 0.5, 1]

            b1 = quantile(control_LLHs, qs); b2 = quantile(drug_LLHs, qs)
            l1 = b1[1:-1]; l2 = b2[1:-1]
            b3 = [-100, -10, -5, -3, -1, 0, 1, 3, 5, 10, 100]
            l3 = b3[1:-1]
            cmap1 = ListedColormap(['navy', 'blue', 'dodgerblue', 'deepskyblue', 'skyblue'])
            norm1 = BoundaryNorm(b1, cmap1.N)
            cmap2 = ListedColormap(['firebrick', 'crimson', 'red', 'orangered', 'darkorange'])
            norm2 = BoundaryNorm(b2, cmap2.N)
            cmap3 = ListedColormap(['firebrick', 'crimson', 'red', 'orangered', 'darkorange', 'skyblue', 'deepskyblue', 'dodgerblue', 'blue', 'navy'])
            norm3 = BoundaryNorm(b3, cmap3.N)

            fig, (l, r, b) = subplots(1, 3, figsize=(16, 4), dpi=300)

            l.set_title('Control'); r.set_title('Drug'); b.set_title('Combined')

            im1 = l.imshow(zg1.T, origin='lower', cmap=cmap1, norm=norm1, extent=(-15, 15, -15, 15))
            im2 = r.imshow(zg2.T, origin='lower', cmap=cmap2, norm=norm2, extent=(-15, 15, -15, 15))
            im3 = b.imshow(zg3.T, origin='lower', cmap=cmap3, norm=norm3, extent=(-15, 15, -15, 15))

            cs1 = l.contour(xg.T, yg.T, zg1.T, levels=l1, colors='k')
            cs2 = r.contour(xg.T, yg.T, zg2.T, levels=l2, colors='k')
            cs3 = b.contour(xg.T, yg.T, zg3.T, levels=l3, colors='k')

            cbar1 = colorbar(im1, ax=l, label='Quantile')
            cbar2 = colorbar(im2, ax=r, label='Quantile')
            cbar3 = colorbar(im3, ax=b, label='Log Prob. Ratio')

            cbar1.ax.set_yticklabels(qs)
            cbar2.ax.set_yticklabels(qs)
            cbar3.ax.set_yticklabels(b3)

            control_below = (self.control_qLLHs < threshold)
            drug_below = (self.drug_qLLHs < threshold)

            l.scatter(flip(self.X_control_reduced)[0,control_below], flip(self.X_control_reduced)[1,control_below], zorder=2, c='white', s=0.5, marker='+')
            l.scatter(flip(self.X_control_reduced)[0,~control_below], flip(self.X_control_reduced)[1,~control_below], zorder=1, c='white', s=0.5)

            r.scatter(flip(self.X_penetramax_reduced)[0,drug_below], flip(self.X_penetramax_reduced)[1,drug_below], zorder=2, c='black', s=0.5, marker='+')
            r.scatter(flip(self.X_penetramax_reduced)[0,~drug_below], flip(self.X_penetramax_reduced)[1,~drug_below], zorder=1, c='black', s=0.5)

            b.scatter(flip(self.X_control_reduced)[0,control_below], flip(self.X_control_reduced)[1,control_below], zorder=2, c='white', s=0.5, marker='+')
            b.scatter(flip(self.X_control_reduced)[0,~control_below], flip(self.X_control_reduced)[1,~control_below], zorder=1, c='white', s=0.5)
            b.scatter(flip(self.X_penetramax_reduced)[0,drug_below], flip(self.X_penetramax_reduced)[1,drug_below], zorder=2, c='black', s=0.5, marker='+')
            b.scatter(flip(self.X_penetramax_reduced)[0,~drug_below], flip(self.X_penetramax_reduced)[1,~drug_below], zorder=1, c='black', s=0.5)

            for ax in [l, r, b]:
                ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
                ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2')

            tight_layout()
            if save_to is not None:
                savefig(save_to, dpi=300, bbox_inches='tight')
            show()

    def makeDBSCAN(self, eps:float=7, min_samples:int=100, show_plot:bool=False, save_to=None):
        from sklearn.cluster import DBSCAN
        from numpy import append
        from matplotlib.pyplot import subplots, tight_layout, savefig, show

        clusters_control = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.X_control)
        clusters_drug = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self.X_penetramax)
        self.df['Cluster'] = append(clusters_control, clusters_drug)

        if show_plot:
            fig, (l, r, b) = subplots(1, 3, figsize=(16, 4), dpi=300)

            l.set_title('Control'); r.set_title('Drug'); b.set_title('Combined')

            l.scatter(self.X_control_reduced[:,0], self.X_control_reduced[:,1], c=['dodgerblue' if c>-1 else 'k' for c in clusters_control])
            r.scatter(self.X_penetramax_reduced[:,0], self.X_penetramax_reduced[:,1], c=['firebrick' if c>-1 else 'k' for c in clusters_drug])
            
            b.scatter(self.X_reduced[self.is_control,0], self.X[self.is_control,1], c=['dodgerblue' if c>-1 else 'k' for c in clusters_control])
            b.scatter(self.X_reduced[~self.is_control,0], self.X[~self.is_control,1], c=['firebrick' if c>-1 else 'k' for c in clusters_control])

            for ax in [l, r, b]:
                ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
                ax.set_xlabel('PCA 1'); ax.set_ylabel('PCA 2')

            tight_layout()
            if save_to is not None:
                savefig(save_to, dpi=300, bbox_inches='tight')
            show()

    def showBestAndWorst(self, save_to=None):
        from numpy import argsort
        from matplotlib.pyplot import subplots, tight_layout, savefig, show

        worst_to_best_control = argsort(self.control_qLLHs)
        worst_to_best_drug = argsort(self.drug_qLLHs)

        kwargs = {'vmin': 0, 'vmax': 255, 'cmap': 'Reds', 'origin': 'lower'}

        fig, axes = subplots(2, 4, figsize=(12, 6), dpi=300)

        axes[0,0].set_title('Worst control')
        axes[0,0].imshow(self.control_images_R[worst_to_best_control[0]], **kwargs)
        axes[0,1].set_title('Second-worst control')
        axes[0,1].imshow(self.control_images_R[worst_to_best_control[1]], **kwargs)
        axes[0,2].set_title('Second-best control')
        axes[0,2].imshow(self.control_images_R[worst_to_best_control[-2]], **kwargs)
        axes[0,3].set_title('Best control')
        axes[0,3].imshow(self.control_images_R[worst_to_best_control[-1]], **kwargs)

        axes[1,0].set_title('Worst drug')
        axes[1,0].imshow(self.drug_images_R[worst_to_best_drug[0]], **kwargs)
        axes[1,1].set_title('Second-worst drug')
        axes[1,1].imshow(self.drug_images_R[worst_to_best_drug[1]], **kwargs)
        axes[1,2].set_title('Second-best drug')
        axes[1,2].imshow(self.drug_images_R[worst_to_best_drug[-2]], **kwargs)
        axes[1,3].set_title('Best drug')
        axes[1,3].imshow(self.drug_images_R[worst_to_best_drug[-1]], **kwargs)

        for ax in axes.flatten():
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        tight_layout()
        if save_to is not None:
            savefig(save_to, dpi=300, bbox_inches='tight')
        show()

    def makeSelectionKDE(self, q_control=0.05, q_drug=0.05):
        condition = (self.df['label'] == 0) * (self.df['Similarity'] >= q_control) + (self.df['label'] == 1) * (self.df['Similarity'] >= q_drug)
        df_reduced = self.df[condition]
        return df_reduced[self.feature_names + ['label']]
    
    def retrieveImages(self, final_size:int=-1):
        from tqdm import tqdm

        self.cell_images = []; self.nucleus_images = []
        self.labels = []
        for mask in tqdm(self.masks):
            images, labels = mask.getImages(final_size=final_size)
            self.cell_images += images[0]
            self.nucleus_images += images[1]
            self.labels += labels
        return (self.cell_images, self.nucleus_images), self.labels

    def createImageDataset(self, final_size:int=-1):
        from os import mkdir
        from os.path import exists
        from PIL import Image
        from pandas import DataFrame
        from tqdm import tqdm
        from pytorch import ImageDataset

        df_dict = {'cell_fname': [], 'nucleus_fname': [], 'label': []}

        try:
            cell_images, nucleus_images, labels = self.cell_images, self.nucleus_images, self.labels
        except:
            (cell_images, nucleus_images), labels = self.retrieveImages(final_size=final_size)

        # Create data directories
            
        cell_control_dir = 'pytorch_dataset/cells/control/'
        cell_drug_dir = 'pytorch_dataset/cells/drug/'
        nucleus_control_dir = 'pytorch_dataset/nuclei/control/'
        nucleus_drug_dir = 'pytorch_dataset/nuclei/drug/' 

        for dir in [cell_control_dir, cell_drug_dir, nucleus_control_dir ,nucleus_drug_dir]:
            if exists(dir) == False:
                mkdir(dir)

        # Save images to paths
                
        for i, (cell_image, nucleus_image, label) in tqdm(enumerate(zip(cell_images, nucleus_images, labels))):
            images1 = flip_and_rotate(cell_image)
            images2 = flip_and_rotate(nucleus_image)

            for j, (im1, im2) in enumerate(zip(images1, images2)):
                fname = f"{i}_{j}.png"

                im1 = Image.fromarray(im1)
                im2 = Image.fromarray(im2)

                if label == 0:
                    cell_dir = cell_control_dir
                    nucleus_dir = nucleus_control_dir
                else:
                    cell_dir = cell_drug_dir
                    nucleus_dir = nucleus_drug_dir

                im1.save(cell_dir + fname)
                im2.save(nucleus_dir + fname)

                df_dict['cell_fname'].append(cell_dir + fname)
                df_dict['nucleus_fname'].append(nucleus_dir + fname)
                df_dict['label'].append(label)

        self.image_df = DataFrame().from_dict(df_dict)

        return ImageDataset(self.image_df, image_type='cell'), ImageDataset(self.image_df, image_type='nucleus')   