"""Engance images using Single-Scale Retinex (SSR), Multi-Scale Retinex (MSR), 
and MSR with Colour Restoration (MSRCR)

References:
    - https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
    - https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
    - https://docs.opencv.org/3.4/d2/de8/group__core__array.html
    - https://docs.cupy.dev/en/stable/reference/index.html
    - https://docs.opencv.org/4.7.0/d6/dc7/group__imgproc__hist.html
"""
import cupy as cp
import logging
import numpy as np
import cv2


def normalise(image:cp.ndarray, mode:any="range"):
    """Normalise full image to 0-255, maximum to 255, or multiply"""
    ## Nomalise min. and max.to 0-255
    if mode == "range":
        ## Mask to ignore -INF elements, and normalise to 0-255
        mask = cp.logical_or(cp.isnan(image),cp.isinf(image))
        min_val = cp.min(image[~mask])
        max_val = cp.max(image[~mask])
        logger.debug(f"range mode")
        return  (image - min_val) / (max_val - min_val) * 255
    ## Normalise max. to 255
    elif mode == "max":
        logger.debug(f"max mode")
        return image/cp.max(image)*255
    ## Multiply
    elif type(mode) == int:
        logger.debug(f"custom mode")
        return image*mode
    ## Skip normalisation
    elif mode == None:
        logger.debug(f"normalising deselected")
        return image
    

def contrast_stretch(image:cp.ndarray, low_percent=1, high_percent=1):
    """Clip the darkest and lightest pixels of an image and stretch to 0-255"""
    image = cp.asnumpy(image).astype(np.uint8)
    ## No. of pixels to clip
    pixelcount = image.shape[0]*image.shape[1]
    low_count = pixelcount*low_percent/100
    high_count = pixelcount*(1 - high_percent/100)
    ## Separate colour channels
    channels = []
    if len(image.shape) == 2:
        ## Single channel image
        channels = [image]
    else:
        ## Multi channel image
        channels = cv2.split(image)
    ## Placeholder for contrast stretched image
    stretched_image = []
    for i,channel in enumerate(channels):
        # logger.debug(f"intensity dtype: {type(intensity[0,0])}")
        cumulative_histogram = np.cumsum(
            cv2.calcHist([channel],[0],None,[256],(0,256))
        )
        low_index,high_index = np.searchsorted(
            cumulative_histogram,
            (low_count,high_count)
        )
        if low_index == high_index:
            ## Skip normalising
            logger.debug(f"contrast not stretched")
            stretched_image.append(channel)
            continue
        lut = np.array(
            [0 if i < low_index 
                else (255 if i > high_index 
                    else round((i - low_index)/(high_index - low_index)*255)) 
                for i in np.arange(0,256)], 
            dtype = 'uint8'
        )
        # logger.debug(f"shape channel: {image[:,:,0].shape}, lut: {lut.shape}")
        # logger.debug(f"length lut: {lut.size}")
        # logger.debug(f"continuity lut: {lut.flags['C_CONTIGUOUS'] or lut.flags['F_CONTIGUOUS']}")
        # logger.debug(f"dtype channel: {type(image[0,0,0])}")
        stretched_image.append(cv2.LUT(channel,lut))
        logger.debug(f"contrast stretched")
    if len(stretched_image) == 1:
        ## Singel channel image
        return np.squeeze(stretched_image)
    elif len(stretched_image) > 1:
        ##Mutli channel image
        return cv2.merge(stretched_image)
    return None


def ssr(image:np.ndarray, sigma:int=80, norm_mode:any=255):
    """Perform Single Scale Retinex on an image"""
    ## from opencv documentation, sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    ksize = int(((sigma - 0.8)/0.15) + 3.0)
    ## Vector(s) corresponding to gaussian 2D kernel
    kernel_vector = cv2.getGaussianKernel(ksize,sigma)
    ## Use filter2D instead of built-in gaussian blur to speed up with FFT
    gaussian_blur = cv2.filter2D(image,
                                 -1,
                                 np.outer(kernel_vector,kernel_vector))
    logger.debug(f"gaussian min:{cp.min(gaussian_blur)}, max: {cp.max(gaussian_blur)}")
    image = cp.array(image)
    gaussian_blur = cp.array(gaussian_blur)
    ## SSR formula (+1.0 to avoid -INF elements from division by 0)
    return normalise(cp.log10(image) - cp.log10(gaussian_blur + 1.0),
                     norm_mode)


def msr(image:np.ndarray,
        sigma_arr:list=[15,80,250],
        msr_norm_mode:any="range",
        ssr_norm_mode:any=255,
        weights:list=None):
    """Perform Multi Scale Retinex on an image"""
    if weights == None:
        weights = cp.array([1]*len(sigma_arr))
    logger.debug(f"weights: {weights}")
    image_msr = cp.zeros_like(image, dtype=np.float64)
    for i,sigma in enumerate(sigma_arr):
        image_msr += weights[i]*ssr(image,sigma,ssr_norm_mode)
    # return image_msr/cp.sum(cp.array(weights))
    return normalise(image_msr/len(weights),msr_norm_mode)


def msrcr(image:np.ndarray,
          sigma_arr:list=[15,80,250],
          low_percent:int=1,
          high_percent:int=1,
          msrcr_norm_mode:any="range",
          msr_norm_mode:any=None,
          ssr_norm_mode:any=None,
          alpha:int=125,
          beta:int=46,
          gain:int=192,
          offset:int=-30,
          weights:list=None):
    """Perform Multi Scale Retinex with Colour Restoration"""
    image = image.astype(np.float64) + 1.0
    image_msr = msr(image,sigma_arr,msr_norm_mode,ssr_norm_mode,weights)
    ## Colour restoration function
    crf = cp.array(
        beta*(np.log10(alpha*image) - np.log10(np.sum(image,2, keepdims=True)))
    )
    ## MSRCR formula
    image_msrcr = gain*(image_msr*crf - offset)
    ## Normalise
    image_msrcr = normalise(image_msrcr,msrcr_norm_mode)
    ## Contrast stretch and return
    image_msrcr = contrast_stretch(image_msrcr,low_percent,high_percent)
    return image_msrcr


## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("retinex")
## Main code
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ## Input/output
    dir_in = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\image-footage"
    dir_out = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Algorithm Outputs"
    filename = "culaneNight_1.png"
    image_in = cv2.imread(f"{dir_in}\\{filename}")
            
    def run_ssr(test:bool=False):
        sigma = 80
        if test:
            ## Vary sigma
            global filename,image_in
            logger.debug(f"input min: {cp.min(image_in)}, max: {cp.max(image_in)}")
            for i in range(1,26):
                logger.debug(f"i: {i}")
                sigma = i*10
                image_out = cp.asnumpy(ssr(image_in,sigma))
                logger.debug(f"masked output min: {np.min(np.ma.masked_invalid(cp.asnumpy(image_out)))}, max: {np.max(np.ma.masked_invalid(cp.asnumpy(image_out)))}")
                cv2.imwrite(f"{dir_out}\\retinex_ssr\\test\\{filename[:-4]}_ssr_sigma{sigma}.png",image_out)
        else:
            ## Batch process
            for i in range(1,13):
                logger.debug(f"i: {i}")
                filename = f"{filename[:-5]}{i}.png"
                image_in = cv2.imread(f"{dir_in}\\{filename}")
                image_out = cp.asnumpy(ssr(image_in,sigma))
                cv2.imwrite(f"{dir_out}\\retinex_ssr\\{filename[:-4]}_ssr.png",image_out)
    
    def run_msr(test:bool=False):
        sigma_arr = [15,80,250]
        if test:
            ## Vary sigma
            global filename,image_in
            image_out = cp.asnumpy(msr(image_in, sigma_arr))
            cv2.imwrite(f"{dir_out}\\retinex_msr\\test\\{filename[:-4]}_msr.png",image_out)
        else:
            for i in range(1,13):
                logger.debug(f"i: {i}")
                filename = f"{filename[:-4]}{i}.png"
                image_in = cv2.imread(f"{dir_in}\\{filename}")
                image_out = cp.asnumpy(msr(image_in, sigma_arr))
                cv2.imwrite(f"{dir_out}\\retinex_msr\\{filename[:-4]}_msr.png",image_out)
    
    def run_msrcr(test:bool=False):
        low_percent = 50
        sigma_arr = [15,80,250]
        # low_percent = 80
        # sigma_arr = [10,20,40,80,160]
        if test:
            for i in range(1,10):
                logger.debug(f"i: {i}")
                low_percent = i*10
                global filename,image_in
                logger.debug(f"input dtype: {image_in.dtype}")
                image_out = msrcr(image_in,sigma_arr,low_percent)
                cv2.imwrite(f"{dir_out}\\retinex_msrcr\\test\\{filename[:-4]}_msrcr_lower{low_percent}.png",image_out)
        else:
            for i in range(1,9):
                logger.debug(f"i: {i}")
                filename = f"{filename[:-5]}{i}.png"
                image_in = cv2.imread(f"{dir_in}\\{filename}")
                image_out = msrcr(image_in,sigma_arr,low_percent)
                cv2.imwrite(f"{dir_out}\\retinex_msrcr\\{filename[:-4]}_msrcr.png",image_out)
                logger.debug(f"{dir_out}\\retinex_msrcr\\{filename[:-4]}_msrcr.png")

    

    # run_ssr(test=False)
    # run_msr(test=False)
    run_msrcr(test=False)
