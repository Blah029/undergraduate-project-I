"""Engance images using Single-Scale Retinex (SSR), Multi-Scale Retinex (MSR), 
MSR with Colour Restoration (MSRCR), or MSR with Colour Preservation (MSRCP)

References:
    - https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
    - https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
    - https://docs.opencv.org/3.4/d2/de8/group__core__array.html
"""
import logging
import numpy as np
import cv2


def normalise(image:np.ndarray, mode:any="range"):
    """Normalise full image to 0-255, maximum to 255, or multiply"""
    ## Mask to ignore -INF elements, and normalise to 0-255
    if mode == "range":
        return (image-np.min(np.ma.masked_invalid(image)))/\
            (np.max(image)-np.min(np.ma.masked_invalid(image)))*255
    ## Normalise to 255
    elif mode == "max":
        return image/np.max(image)*255
    ## Multiply
    elif type(mode) == int:
        return image*mode


def ssr(image:np.ndarray, sigma:int=100, ksize:int=0, norm_mode:any=255):
    """Perform Single Scale Retinex on an image"""
    if ksize == 0:
        ## from opencv documentation, sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
        ksize = int(((sigma - 0.8)/0.15) + 3.0)
    ## Vector(s) corresponding to gaussian 2D kernel
    kernel_vector = cv2.getGaussianKernel(ksize,sigma)
    ## Use filter2D instead of built-in gaussian blur to speed up with FFT
    gaussian_blur = cv2.filter2D(image,
                                 -1,
                                 np.outer(kernel_vector,kernel_vector))
    logger.debug(f"gaussian min:{np.min(gaussian_blur)}, max: {np.max(gaussian_blur)}")
    ## SSR formula (+1.0 to avoid -INF elements from division by 0)
    return normalise(np.log10(image) - np.log10(gaussian_blur + 1.0),norm_mode)


def msr(image:np.ndarray, 
        sigma_arr:list=[10,100,200], 
        weights:list=None,
        norm_mode:any=255):
    """Perform Multi Scale Retinex on an image"""
    if weights == None:
        weights = [1]*len(sigma_arr)
    logger.debug(f"weights: {weights}")
    image_msr = np.zeros_like(image)
    for i,sigma in enumerate(sigma_arr):
        image_msr = image_msr + weights[i]*\
            ssr(image,sigma, norm_mode=norm_mode)
    return image_msr/len(sigma_arr)


## Set up the logger
logging.basicConfig(format="[%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger("retinex")
## Main code
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    ## Input/output
    dir_in = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Datasets\\image-footage"
    dir_out = "C:\\Users\\User Files\\Documents\\University\\Misc\\4th Year Work\\Final Year Project\\Outputs\\Algorithm Outputs"
            
    def run_ssr(test:bool=False):
        if test:
            ## Vary sigma
            filename = "techodyssey_1080p_1.png"
            image_in = cv2.imread(f"{dir_in}\\{filename}")
            logger.debug(f"input min: {np.min(image_in)}, max: {np.max(image_in)}")
            norm_mode = "range"
            for i in range(1,51):
                sigma = i*10
                image_out = ssr(image_in,sigma, norm_mode=norm_mode)
                logger.debug(f"retinex min: {np.min(image_out)}, max: {np.max(image_out)}")
                logger.debug(f"masked min: {np.min(np.ma.masked_invalid(image_out))}, max: {np.max(np.ma.masked_invalid(image_out))}")
                # image_out = normalise_255(image_out)
                cv2.imwrite(f"{dir_out}\\retinex_ssr\\test\\{filename[:-4]}_ssr_sigma{sigma}.png",image_out)
        else:
            ## Batch process
            # sigma = 30
            # norm_mode = 512 + 128
            sigma = 100
            norm_mode = "range"
            for i in range(1,4):
                filename = f"sample_night_{i}.png"
                image_in = cv2.imread(f"{dir_in}\\{filename}")
                image_out = ssr(image_in,sigma, norm_mode=norm_mode)
                cv2.imwrite(f"{dir_out}\\retinex_ssr\\{filename[:-4]}_ssr.png",image_out)
    
    def run_msr(test:bool=False):
        if test:
            filename = "techodyssey_1080p_1.png"
            image_in = cv2.imread(f"{dir_in}\\{filename}")
            image_out = msr(image_in,[10,40,80], norm_mode=512+128)
            cv2.imwrite(f"{dir_out}\\retinex_msr\\test\\{filename[:-4]}_msr.png",image_out)
    
    run_ssr(test=False)
    # run_msr(test=True)
