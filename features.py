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

def segmental_correction(image, masks, flux_normalisation=gaussian_normalisation):
    from liams_funcs import decompose_masks
    from numpy import stack, nan, ones_like

    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    R1, G1 ,B1 = nan * ones_like(R), nan * ones_like(G), nan * ones_like(B)
    for mask in decompose_masks(masks, reduce=False):
        R1[mask] = flux_normalisation(R[mask])
        G1[mask] = flux_normalisation(G[mask])
        B1[mask] = flux_normalisation(B[mask])

    return stack([R1 ,G1, B1], axis=2)

def erodeNtimes(mask, N=1):
    from skimage.morphology import binary_erosion
    if N == 1: return binary_erosion(mask)
    return erodeNtimes(binary_erosion(mask), N=N-1)

def ZtoRGB(channel, offset:int=1, min=None, max=None):
    from numpy import uint8, nanmax, nanmin, where, isfinite
    if min is None: min = nanmin(channel)
    if max is None: max = nanmax(channel)
    return uint8(where(isfinite(channel), offset + (255-offset) * (channel - min) / (max - min), 0))

### Classes

class Segment:
    def __init__(self, cell_rp, nucleus_rp, segment_order:int=0, min=None, max=None):
        from numpy import where, nan, quantile, clip, exp, pi
        from mahotas.features import haralick
        from pathtest import main_pathAnalysis
        from Functions_George import detect_outline
        from skimage.filters import gaussian

        self.cell_rp = cell_rp
        self.nucleus_rp = nucleus_rp
        self.segment_order = segment_order

        # Colour channels
        self.cellR = where(self.cell_rp.image, self.cell_rp.intensity_image[:,:,0], nan)
        self.cellB = where(self.cell_rp.image, self.cell_rp.intensity_image[:,:,2], nan)
        self.nucleusR = where(self.nucleus_rp.image, self.nucleus_rp.intensity_image[:,:,0], nan)
        self.nucleusB = where(self.nucleus_rp.image, self.nucleus_rp.intensity_image[:,:,2], nan)

        # Binary channels
        self.binary_z = lambda z: clip((gaussian(self.nucleusR) > z).astype(int) + detect_outline(self.nucleus_rp.image), 0, 1)
        self.binary_q = lambda q: clip((gaussian(self.nucleusR) > quantile(self.nucleusR[self.nucleus_rp.image], q)).astype(int) + detect_outline(self.nucleus_rp.image), 0, 1)

        self.binary_q50 = self.binary_q(0.5)

        # Haralick features

        self.haralickR = haralick(gaussian(ZtoRGB(self.nucleusR, min=min, max=max), preserve_range=True), ignore_zeros=True).mean(0)
        self.haralickB = haralick(gaussian(ZtoRGB(self.nucleusB, min=min, max=max), preserve_range=True), ignore_zeros=True).mean(0)

        # Possible paths

        #self.paths = main_pathAnalysis(self.binary_q50)

        # Gaussian distribution

        self.gaussian = lambda z: exp(-z**2) / (2 * pi)**0.5
        

    def show_segments(self, crange=5):
        from matplotlib.pyplot import subplots, tight_layout, show, colorbar

        vmin, vmax = -crange, crange
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


    #def getTotalPaths(self): return self.paths[0]
    #def getPathLength(self): return self.paths[1]
    
    ### Retrieving methods

    def retrieveFeatureNames(self): return [func for func in dir(self) if callable(getattr(self, func)) and func.startswith("get")]

    def retrieveFeatures(self): return [getattr(self, func)() for func in self.retrieveFeatureNames()]

class Masks:
    def __init__(self, fname, type='control', flux_normalisation=gaussian_normalisation, min=None, max=None):
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
        for cell_rp, nucleus_rp in zip(self.cell_regionprops, self.nucleus_regionprops):
            self.segments.append(Segment(cell_rp, nucleus_rp, min=min, max=max))

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