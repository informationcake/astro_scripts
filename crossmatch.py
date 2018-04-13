import os, sys, pickle
import numpy as np
import pyvo as vo
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy import units as u
from astropy.coordinates import ICRS, match_coordinates_sky, Angle
from astropy.io import fits, ascii
from astropy.table import Table, hstack, join

######################################################################
############################# Functions ##############################
######################################################################

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def mean_weight(c1,c2,e1,e2):
    diff = (c1-c2)
    norm_weighted_diff = np.sum(diff / (e1**2+e2**2)) / np.sum(1/(e1**2+e2**2))
    return np.average(diff), norm_weighted_diff

def convert_to_fits(msss_cat):
    #if .txt file do this, but need to remove comments at start of file, not sure why...
    rdr = ascii.get_reader(Reader=ascii.Basic)
    rdr.header.splitter.delimiter = ' '
    rdr.data.splitter.delimiter = ' '
    rdr.header.start_line = 0
    rdr.data.start_line = 1
    rdr.data.end_line = None
    rdr.header.comment = r'\s*#'
    rdr.data.comment = r'\s*#'
    data=rdr.read(msss_cat+'.txt')
    data.write(msss_cat+'.fits', format='fits')

def match_TGSS_catalogue(TGSS_cat, MSSS_cat, mosaic_name, inverse_match=False):
    if inverse_match==False:
        c_tgss = ICRS(ra=Angle(TGSS_cat['RA'],unit=u.deg), dec=Angle(TGSS_cat['DEC'],unit=u.deg))
        c_msss = ICRS(ra=Angle(MSSS_cat['RA'],unit=u.deg), dec=Angle(MSSS_cat['DEC'],unit=u.deg))
        idx, d2d, d3d = match_coordinates_sky(c_msss, c_tgss) #matches. idx in second one matching first.
        tbdata=hstack([Table(MSSS_cat), Table(TGSS_cat[idx])], join_type='exact') #join tables together
        return tbdata
    if inverse_match==True:
        #crossmatch TGSS to MSSS, so first grab only small TGSS area
        pos=(int(mosaic_name[1:4]), int(mosaic_name[5:7]))
        service = vo.dal.SCSService("http://vo.astron.nl/tgssadr/q/cone/scs.xml?")
        print('querying TGSS catalogue through PYVO around {0} ...'.format(pos))
        resultset = service.search(pos=pos, radius=5, verbosity=0)
        TGSS_cat=resultset.table
        #Convert TGSS units to arcseconds and keep column names consistent to MSSS, because PYVO query has different column names to downlodable catalogue from website.
        TGSS_cat['e_DEC']/=3600
        TGSS_cat['e_RA']/=3600
        TGSS_cat['e_RA'].name='E_RA'
        TGSS_cat['e_DEC'].name='E_DEC'
        #TGSS_cat['RA']+=0.5 #offset to test scatter for forced no matches
        #prep data for crossmatching
        c_tgss = ICRS(ra=Angle(TGSS_cat['RA'],unit=u.deg), dec=Angle(TGSS_cat['DEC'],unit=u.deg))
        c_msss = ICRS(ra=Angle(MSSS_cat['RA'],unit=u.deg), dec=Angle(MSSS_cat['DEC'],unit=u.deg))
        idx, d2d, d3d = match_coordinates_sky(c_tgss, c_msss) #find closest MSSS source for each TGSS source.
        tbdata=hstack([Table(TGSS_cat), Table(MSSS_cat[idx])], join_type='exact') #join tables together.
        return tbdata

def match_NVSS_catalogue(NVSS_cat, MSSS_cat, mosaic_name):
    #currently ALWAYS match MSSS to NVSS, not other way around
    pos=(int(mosaic_name[1:4]), int(mosaic_name[5:7]))
    c_nvss = ICRS(ra=Angle(NVSS_cat['RA'],unit=u.deg), dec=Angle(NVSS_cat['DEC'],unit=u.deg))
    c_msss = ICRS(ra=Angle(MSSS_cat['RA'],unit=u.deg), dec=Angle(MSSS_cat['DEC'],unit=u.deg))
    idx, d2d, d3d = match_coordinates_sky(c_msss, c_nvss) #matches. idx in second one matching first.
    tbdata=hstack([Table(MSSS_cat), Table(NVSS_cat[idx])], join_type='exact') #join tables together
    tbdata['RA_ERROR'].name='E_RA_2' #rename columns consistently to MSSS
    tbdata['DEC_ERROR'].name='E_DEC_2'
    tbdata['E_RA'].name='E_RA_1'
    tbdata['E_DEC'].name='E_DEC_1'
    tbdata['E_RA_2']/=3600 #convert to degrees
    tbdata['E_DEC_2']/=3600
    return tbdata

def plot_rotation_test(ra1, dec1, diff_RA, diff_DEC, sig, mosaic_name, matched_cat, inverse_match=False):
    #test for rotation; is offset_RA a function of DEC?
    fig=plt.figure(figsize=(10,6))
    if matched_cat=='TGSS':
        if inverse_match==False:
            fig.suptitle('{0} matched to TGSS.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'MSSStoTGSS_RotationTest.png'
            fig.text(0.04, 0.5, 'Offset of MSSS relative to TGSS, arcseconds', va='center', rotation='vertical')
        else:
            fig.suptitle('TGSS matched to {0}.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'TGSStoMSSS_RotationTest.png'
            fig.text(0.04, 0.5, 'Offset of TGSS relative to MSSS, arcseconds', va='center', rotation='vertical')
    if matched_cat=='NVSS':
        if inverse_match==False:
            fig.suptitle('{0} matched to NVSS.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'MSSStoNVSS_RotationTest.png'
            fig.text(0.04, 0.5, 'Offset of MSSS relative to NVSS, arcseconds', va='center', rotation='vertical')
        else:
            fig.suptitle('NVSS matched to {0}.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'NVSStoMSSS_RotationTest.png'
            fig.text(0.04, 0.5, 'Offset of NVSS relative to MSSS arcseconds', va='center', rotation='vertical')
    plt.subplot(2, 1, 1)
    plt.scatter(ra1, diff_RA*3600, s=1, label='RA')
    plt.scatter(ra1, diff_DEC*3600, s=1, label='DEC', c='r')
    plt.axhline(y=0, ls='-', lw=0.5, color='k')
    plt.xlabel('RA of MSSS source')
    plt.legend(frameon=True, numpoints=1)
    plt.subplot(2, 1, 2)
    plt.scatter(dec1, diff_RA*3600, s=1, label='RA')
    plt.scatter(dec1, diff_DEC*3600, s=1, label='DEC', c='r')
    plt.axhline(y=0, ls='-', lw=0.5, color='k')
    plt.xlabel('DEC of MSSS source')
    plt.legend(frameon=True, numpoints=1)
    plt.savefig(title)
    #plt.show()

def plot_offsets_radar(diff_RA, diff_DEC, new_RA_offset, new_DEC_offset, sig, init_sources, final_sources, mosaic_name, matched_cat='TGSS', inverse_match=False):
    # Create the radar/target plot of offsets. Ellipses not working yet?
    #cov = np.cov(diff_RA, diff_DEC)
    #var, lamb = np.linalg.eig(cov)
    #stdev = np.sqrt(var)
    #stdev2 = stdev
    #stdev2 = np.sqrt(np.var(offset))
    # Plot the RA/DEC offsets
    fig=plt.figure(figsize=(5,5))
    if matched_cat=='TGSS':
        if inverse_match==False:
            fig.suptitle('{0} matched to TGSS.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'MSSStoTGSS_RadarPlot.png'
        else:
            fig.suptitle('TGSS matched to {0}.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'TGSStoMSSS_RadarPlot.png'
    if matched_cat=='NVSS':
        if inverse_match==False:
            fig.suptitle('{0} matched to NVSS.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'MSSStoNVSS_RadarPlot.png'
        else:
            fig.suptitle('NVSS matched to {0}.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'NVSStoMSSS_RadarPlot.png'
    #ax = fig.add_subplot(111)
    #plt.scatter(diff_RA, diff_DEC, color='k', marker='o',s=1)
    plt.errorbar(diff_RA*3600, diff_DEC*3600, xerr=errorsRA*3600, yerr=errorsDEC*3600, color='k', ls=' ', marker='o', ms=1, capsize=None, elinewidth=0.5)
    '''
    for j in range(1, 4):
        ell = Ellipse(xy=(np.mean(diff_RA), np.mean(diff_DEC)),
            width=stdev2*j*2, height=stdev2*j*2,
            angle=np.rad2deg(np.arccos(lamb[0, 0])))
        ell.set_facecolor('none')
        plt.gca().add_patch(ell)
    '''
    plt.xlabel('RA offset, arcseconds')
    plt.ylabel('DEC offset, arcseconds')
    plt.axhline(y=np.mean(diff_DEC*3600), ls='--', lw=0.5, color='k')
    plt.axvline(x=np.mean(diff_RA*3600), ls='--', lw=0.5, color='k')
    plt.text(0.05, 0.95, '{0:.2f} +- {1:.2f} \n{2:.2f} +- {3:.2f} \n{4}/{5} sources matched.'.format(new_RA_offset[1]*3600, np.std(diff_RA*3600), new_DEC_offset[1]*3600, np.std(diff_DEC*3600), final_sources, init_sources), horizontalalignment='left',
    verticalalignment='top', transform = plt.gca().transAxes)
    plt.savefig(title, dpi=300, bbox_inches='tight')
    #plt.show()

def plot_offsets(diff_RA, diff_DEC, errorsRA, errorsDEC, RA_offset, DEC_offset, new_RA_offset, new_DEC_offset, sig, mosaic_name, matched_cat='TGSS', inverse_match=False):
    fig=plt.figure()
    if matched_cat=='TGSS':
        if inverse_match==False:
            fig.suptitle('{0} matched to TGSS.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'MSSStoTGSS_offset.png'
        else:
            fig.suptitle('TGSS matched to {0}.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'TGSStoMSSS_offset.png'
    if matched_cat=='NVSS':
        if inverse_match==False:
            fig.suptitle('{0} matched to NVSS.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'MSSStoNVSS_offset.png'
        else:
            fig.suptitle('NVSS matched to {0}.\nSources more than {1} sigma away from mean removed.'.format(str(mosaic_name), sig))
            title=mosaic_name[:7]+'NVSStoMSSS_offset.png'
    plt.errorbar(range(len(dec1)), diff_RA*3600, yerr=errorsRA*3600, ls='', marker='o',c='k',markeredgecolor='k',label='RA')
    plt.errorbar(range(len(dec1)), diff_DEC*3600, yerr=errorsDEC*3600, ls='', marker='o',c='r', markeredgecolor='r', label='DEC')
    plt.axhline(y=new_RA_offset[1]*3600, xmin=0, xmax=100, linewidth=0.7, color = 'k', label='weighted mean before {0} sig clip'.format(sig))
    plt.axhline(y=new_DEC_offset[1]*3600, xmin=0, xmax=100, linewidth=0.7, color = 'r')
    plt.axhline(y=RA_offset[1]*3600, xmin=0, xmax=100, linewidth=1, color = 'k',ls=':', label='weighted mean after {0} sig clip'.format(sig))
    plt.axhline(y=DEC_offset[1]*3600, xmin=0, xmax=100, linewidth=1, color = 'r',ls=':')
    plt.xlabel('source label')
    plt.ylabel('offset, arcseconds')
    plt.legend(frameon=True, numpoints=1)
    #plt.legend(loc=7, frameon=True, numpoints=1)
    plt.savefig(title, dpi=300, bbox_inches='tight')
    #plt.show()

def calculate_offset(tbdata, sig):
    RA_offset=mean_weight(tbdata['RA_1'],tbdata['RA_2'],tbdata['E_RA_1'],tbdata['E_RA_2'])
    DEC_offset=mean_weight(tbdata['DEC_1'],tbdata['DEC_2'],tbdata['E_DEC_1'],tbdata['E_DEC_2'])
    print('number of catalogued sources originally: {0}'.format(len(tbdata)))
    print('ra offset before cut is: {0:.3f} arcseconds'.format(RA_offset[0]*3600))
    print('dec offset before cut is: {0:.3f} arcseconds'.format(DEC_offset[0]*3600))
    meanRA=np.full((len(tbdata),1),RA_offset[1]*3600)
    meanDEC=np.full((len(tbdata),1),DEC_offset[1]*3600)
    newtb=[]
    ra_sig=[]
    dec_sig=[]
    newra1=[]
    newra2=[]
    newdec1=[]
    newdec2=[]
    newra1e=[]
    newra2e=[]
    newdec1e=[]
    newdec2e=[]
    #Now see if matched sources are within sigma limit, remove if not:
    for i in range(0,len(tbdata)):
       ra_test=(RA_offset[1] - (tbdata['RA_1'][i] - tbdata['RA_2'][i])) / np.sqrt(tbdata['E_RA_1'][i]**2+tbdata['E_RA_2'][i]**2) #calculate how many std away it is
       if ra_test<sig:
          ra_sig.append(ra_test)
       dec_test=(DEC_offset[1] - (tbdata['DEC_1'][i] - tbdata['DEC_2'][i])) / np.sqrt(tbdata['E_DEC_1'][i]**2+tbdata['E_DEC_2'][i]**2)
       if dec_test<sig:
          dec_sig.append(dec_test)
       if abs(ra_test)<sig and abs(dec_test)<sig:
          newra1.append(tbdata['RA_1'][i])
          newra2.append(tbdata['RA_2'][i])
          newdec1.append(tbdata['DEC_1'][i])
          newdec2.append(tbdata['DEC_2'][i])
          newra1e.append(tbdata['E_RA_1'][i])
          newra2e.append(tbdata['E_RA_2'][i])
          newdec1e.append(tbdata['E_DEC_1'][i])
          newdec2e.append(tbdata['E_DEC_2'][i])
    #New arrays have only sources within sigma limit
    ra1=np.array(newra1)
    ra2=np.array(newra2)
    dec1=np.array(newdec1)
    dec2=np.array(newdec2)
    ra1e=np.array(newra1e)
    ra2e=np.array(newra2e)
    dec1e=np.array(newdec1e)
    dec2e=np.array(newdec2e)

    new_RA_offset=mean_weight(ra1,ra2,ra1e,ra2e)
    new_DEC_offset=mean_weight(dec1,dec2,dec1e,dec2e)
    print('Final number of sources after discarding sources {0} sigma away from mean: {1}'.format(sig, len(ra1)))
    print('New ra offset after cut is: {0:.3f}'.format(new_RA_offset[1]*3600))
    print('New dec offset after cut is: {0:.3f}'.format(new_DEC_offset[1]*3600))
    print("-"*50)
    #Gather errors and offsets after sigma cut:
    errorsDEC=np.sqrt((dec1e**2+dec2e**2))
    errorsRA=np.sqrt((ra1e**2+ra2e**2))
    diff_RA=ra1-ra2
    diff_DEC=dec1-dec2
    init_sources=len(tbdata)
    final_sources=len(ra1)
    return diff_RA, diff_DEC, RA_offset, DEC_offset, new_RA_offset, new_DEC_offset, ra1, dec1, errorsRA, errorsDEC, init_sources, final_sources





######################################################################
######################### Beginning Script ###########################
######################################################################

####README####
#ONLY THING YOU NEED TO CHANGE IS THE FLAG FOR MATCHING TO EITHER TGSS OR NVSS, AND THE NUMBER OF SIGMA MATCHES ARE ACCEPTABLE TO WITHIN

if __name__ == "__main__":
    #convert manually for now, automate if I do this for entire survey.
    #msss_cat='M172+53_catalog'
    #convert_to_fits(msss_cat)

    #Which catalogue are you matching with?
    matched_cat='NVSS'
    #matched_cat='TGSS'
    inverse_match=False #Match MSSS to another survey
    #inverse_match=True #Match another survey to MSSS
    #when inverse_match==True, TGSS will struggle to find good matches since the resolution is much better. Need to change sigma to 5 or 10.
    sig=3
    #only keep sources where their offset is this many standard deviations away from the mean.
    #for NVSS this needs to be one or two, since MSSS source density is much higher. For TGSS you can use 5 and get good results, since source densities are comparable-ish.

    #Local files required:
    #msss_filename='M172+53_catalog.fits'
    #msss_filename='M188+53_catalog.fits'
    msss_filename='M205+53_catalog.fits'
    #msss_filename = sys.argv[1]
    msss_fits = fits.open(msss_filename)
    MSSS_cat = msss_fits[1].data

    tgss_filename='TGSSADR1_7sigma_catalog.fits'
    tgss_fits = fits.open(tgss_filename)
    TGSS_cat = tgss_fits[1].data
    #Convert TGSS units to arcseconds and keep column names consistent to MSSS
    TGSS_cat['E_DEC']/=3600
    TGSS_cat['E_RA']/=3600

    nvss_filename='NVSS_cat_alex.fits'
    nvss_fits = fits.open(nvss_filename)
    NVSS_cat = nvss_fits[1].data

    #Shift TGSS sources to create no matches. For the purpose of assessing scatter in plots:
    #TGSS_cat['RA']+=0.5 #Check match_catalogue function to do this when querying pyvo.

    if inverse_match==False:
        print('Find sources in {0} that match  sources in {1}. Using a {2} sigma cut'.format(matched_cat, msss_filename, sig))
    if inverse_match==True:
        print('Find sources in {0} that match  sources in {1}. Using a {2} sigma cut'.format(msss_filename, matched_cat, sig))

    #Finds the closest match. Later those offsets that are farther than (e.g.) 3 sigma away are moreved. To me this is a more rigorous way than specifying in the crossmatching to only match sources within x arcseconds. Because, how do you choose x consistently with different surveys at diff resolutions and depths etc?
    if matched_cat=='TGSS':
        tbdata=match_TGSS_catalogue(TGSS_cat, MSSS_cat, msss_filename, inverse_match=inverse_match)
    if matched_cat=='NVSS':
        tbdata=match_NVSS_catalogue(NVSS_cat, MSSS_cat, msss_filename) #currently no flag to do reverse match.

    #Calculate the offsets, remove sources more than certain number of standard deviations away from mean, before recalculating the offset. This should filter out sources which were false matches, before calculating the offset of real truely matched sources.
    diff_RA, diff_DEC, RA_offset, DEC_offset, new_RA_offset, new_DEC_offset, ra1, dec1, errorsRA, errorsDEC, init_sources, final_sources = calculate_offset(tbdata, sig)

    #Create plots
    plot_offsets(diff_RA, diff_DEC, errorsRA, errorsDEC, RA_offset, DEC_offset, new_RA_offset, new_DEC_offset, sig, msss_filename, matched_cat=matched_cat, inverse_match=inverse_match)
    plot_rotation_test(ra1, dec1, diff_RA, diff_DEC, sig, msss_filename, matched_cat=matched_cat, inverse_match=inverse_match)
    plot_offsets_radar(diff_RA, diff_DEC, new_RA_offset, new_DEC_offset, sig, init_sources, final_sources, msss_filename, matched_cat=matched_cat, inverse_match=inverse_match)



    plt.show()



    #
