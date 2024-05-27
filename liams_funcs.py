def show_image(fname, save_to=None):
    """
    Show an image, along with summed red and blue emission along each axis. 
    """
    from matplotlib.pyplot import imread, subplots, delaxes, subplots_adjust, show, savefig
    from numpy import arange

    im = imread(fname)
    Nx, Ny, _ = im.shape
    
    red = im[:,:,0]
    blue = im[:,:,2]

    fig, axes = subplots(2, 2, gridspec_kw={'width_ratios': [4,1], 'height_ratios': [4,1]}, dpi=300)
    delaxes(axes[1,1])
    subplots_adjust(hspace=0, wspace=0)

    main, bottom, right = axes[0,0], axes[1,0], axes[0,1]

    main.imshow(im, origin='lower', aspect='auto')
    main.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    main.set_ylabel(r'0$^\text{th}$ axis'); main.set_xlabel(r'1$^\text{st}$ axis')
    main.xaxis.set_label_position('top')

    bottom.fill_between(arange(Ny), red.sum(axis=0), color='r', alpha=0.5); bottom.fill_between(arange(Ny), blue.sum(axis=0), color='b', alpha=0.5)
    bottom.plot(arange(Ny), red.sum(axis=0), color='r'); bottom.plot(arange(Ny), blue.sum(axis=0), color='b')

    right.fill_betweenx(arange(Nx), red.sum(axis=1), color='r', alpha=0.5); right.fill_betweenx(arange(Nx), blue.sum(axis=1), color='b', alpha=0.5)
    right.plot(red.sum(axis=1), arange(Nx), color='r'); right.plot(blue.sum(axis=1), arange(Nx), color='b')

    bottom.set_yticks([]); bottom.set_xlim(0, Ny); bottom.set_ylim(0)
    right.set_xticks([]); right.set_ylim(0, Nx); right.set_xlim(0); right.invert_xaxis()
    right.tick_params(labelleft=False, right=True, labelright=True)

    if save_to is not None:
        savefig(save_to, dpi=300, bbox_inches='tight')

    show()

###

def perform_hist_equalisation(image, nbins=256):
    from skimage.exposure import equalize_hist
    from numpy import stack

    red_corrected = equalize_hist(image[:,:,0], nbins=nbins)
    blue_corrected = equalize_hist(image[:,:,2], nbins=nbins)

    return stack([red_corrected, blue_corrected, blue_corrected], axis=2)

def perform_adapthist_equalisation(image, nbins=256):
    from skimage.exposure import equalize_adapthist
    from numpy import stack

    red_corrected = equalize_adapthist(image[:,:,0], nbins=nbins)
    blue_corrected = equalize_adapthist(image[:,:,2], nbins=nbins)

    return stack([red_corrected, blue_corrected, blue_corrected], axis=2)

###

def remove_borders(bool_matrix, flexibility:int=5):
    from numpy import argwhere
    args1 = argwhere(bool_matrix.sum(axis=1) != 0)
    args2 = argwhere(bool_matrix.sum(axis=0) != 0)

    x0, x1 = args1.min()-flexibility, args1.max()+1+flexibility
    y0, y1 = args2.min()-flexibility, args2.max()+1+flexibility

    return bool_matrix[x0:x1,y0:y1]

def decompose_masks(masks, reduce:bool=True, flexibility=5):
    from numpy import arange
    labels = arange(1, int(masks.max()+1), 1)

    result = []
    for label in labels:
        if reduce:
            result.append(remove_borders(masks == label, flexibility=flexibility))
        else:
            result.append(masks == label)
        
    return result

def evaluate_roundness(mask):
    from skimage.measure import perimeter
    from math import pi

    area = mask.sum()
    perim = perimeter(mask)

    return (4 * pi * area) / perim**2

def evaluate_colour(image, mask, bins=10):
    from numpy import histogram

    reds = image[:,:,0][mask] / 255
    blues = image[:,:,2][mask] / 255

    vals_red, edges = histogram(reds, range=(0,1), bins=bins, density=True)
    vals_blue, _ = histogram(blues, range=(0,1), bins=bins, density=True)

    return edges, vals_red, vals_blue

###