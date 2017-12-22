""" Functions to subtract sky:
"""
import pdb

def median_ss(sky_file, img_file, mask_limit, mask_file, ss_out_file, sky_out_file, cut_ch=300):
    """ This function is used to subtract sky median
    Parameters
    ----------
    sky_file
    img_file
    mask_limit : mask limit
    cut_ch     : cut bad channel number
    Eaxmple:
    median_ss('kb171021_00083_icuber.fits', 'kb171021_00082_icuber.fits', 8, cut_ch=300)
    Returns
    -------
    mask.fits  : masked image (mask the bad pixels and qso)
    img.fits   : sky median subtracted image
    """
    from astropy.io import fits
    import numpy as np
    from astropy.stats import sigma_clip
    sky_input, sh = fits.getdata(sky_file, header=True)
    img_input, ih = fits.getdata(img_file, header=True)

    # goodwave layers, default clip 300 layers at the beginning and the end
    img=img_input[cut_ch:(len(img_input[:,1,1])-cut_ch),:,:]
    sky=sky_input[cut_ch:(len(img_input[:,1,1])-cut_ch),:,:]
    sky_cube=sky_input[cut_ch:(len(img_input[:,1,1])-cut_ch),:,:]
    
    ### begin to mask qso on sky
    # define final mask image
    mask_sum = np.zeros(sky[1,:,:].shape)

    ## looping over the layers
    for j in np.arange(0, len(sky[:,1,1]), 1):
        # iterated 5 times, rejecting points by > 3 sigma
        filtered_data = sigma_clip(sky[j,:,:], sigma=3, iters=3)
        sky_value = np.median(filtered_data.data[~filtered_data.mask])
        # subtract the sky median value
        img[j,:,:]-=sky_value
        # save the sky
        sky_cube[j,:,:]=sky_value
        # record the masked counts
        mask = np.zeros(sky[1,:,:].shape)
        mask[filtered_data.mask == True] = 1
        mask_sum+=mask

    # use mask_limit to avoid masking the good pixels (noise)
    mask_sum[mask_sum<mask_limit]=0
    mask_sum[mask_sum>=mask_limit]=1

    fits.writeto(mask_file,mask_sum, clobber=True)
    fits.writeto(ss_out_file, img, ih, clobber=True)
    fits.writeto(sky_out_file,sky_cube,sh,clobber=True)
    #pdb.set_trace()
    return sky_value
    




def psf_sub(img_file,out_file, dpxl, dpxr, dpy, scale):
    """ This function is used to subtract PSF for bright point source
    Parameters
    ----------
    img_file  :   default 'img.fits'
    dpx       :   cut PSF region in x
    dpy       :   cut PSF region in y
    scale     :   sclae method: 'peak', 'p', 'integrated', 'i'
    Eaxmple:
    psf('img.fits', dpx=8, dpy=10, scale='integrated')
    Returns
    -------
    img_psf.fits  : psf subtracted image
    """
    from astropy.io import fits
    import numpy as np
    img, ih = fits.getdata(img_file, header=True)

    ### PSF reduced
    img_median= np.median(img, axis=0)

    # find peak location
    loc = np.where(img_median > 0.9*np.amax(img_median))
    px= int(round(np.mean(loc[1])))
    py= int(round(np.mean(loc[0])))

    a = np.zeros(img[:,1,1].shape)
    psf = np.zeros(img.shape)

    ## scale to each channel
    if scale not in ['peak', 'p', 'integrated', 'i']:
        raise IOError('Need input the scale method! scale = peak / integrated')
    # peak value to scale
    if scale in ['peak', 'p']:
        peak_value = np.max(img_median)
        a=img[:,py,px]/peak_value

    # cut the region
    for j in np.arange(0, len(img[:,1,1]), 1):
        # integrated to scale
        if scale in ['integrated', 'i']:
            integrated_value = np.sum(img[j,(py-dpy):(py+dpy),(px-dpxl):(px+dpxr)])
            a[j]=integrated_value/np.sum(img_median[(py-dpy):(py+dpy),(px-dpxl):(px+dpxr)])
        psf[j,(py-dpy):(py+dpy),(px-dpxl):(px+dpxr)]=img_median[(py-dpy):(py+dpy),(px-dpxl):(px+dpxr)]*a[j]*1.0

    # reduce psf
    img-=psf

    fits.writeto(out_file,img, ih,clobber=True)
