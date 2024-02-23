#! /usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

# ploting option
plt.rcParams['figure.figsize'] = [12.94, 8]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
#plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10

SAVE_FIGS = True

# path to folder containing csv files
SUBDIR = '../unsoda/csv/'

# if one soil data are studied
CODE = 1011

# read data files
df_soil = pd.read_csv(SUBDIR + "soil_properties.csv")
# sat permeability from cm/d to m/s
df_soil.k_sat = df_soil.k_sat.apply(lambda x: x/86400/100)

df_gsd = pd.read_csv(SUBDIR + "particle_size.csv")
# particle_size from µm to mm
df_gsd.particle_size = df_gsd.particle_size.apply(lambda x: x/1000)

df_wrc = pd.read_csv(SUBDIR + "lab_drying_h-t.csv")
# preshead from cm_H20 to kPa
df_wrc.preshead = df_wrc.preshead.apply(lambda x: x/10) 

#print('Soil properties:')
#print(df_soil[df_soil.code == CODE])


# correlation between basic variables
# df_soil.corr()
import seaborn as sns    # Allows for easy plotting of heatmaps
sns_plt = sns.heatmap(df_soil.corr(), annot=True, annot_kws={"fontsize":12})
if SAVE_FIGS:
    plt.savefig("fig_corr.png")
    
else:
    plt.show()
plt.close()

# NB : not enough data for EC to say something about high correlation with k_sat

# porosity vs permeability
df_soil.plot(x="porosity", y="k_sat", style='ro', logy=True, alpha=0.2,
    legend=False, xlabel="Porosity (-)", ylabel="Saturated permeability (m/s)")
if SAVE_FIGS:
    plt.savefig("n-k_sat.pdf")
else:
    plt.show()
plt.close()

# plotting gsd for CODE
#df_gsd[df_gsd.code == CODE].plot(x="particle_size", y="particle_fraction", logx=True, style='ro',
#                                 legend=False, xlabel="Particle size (mm)", ylabel="Fraction (-)")
#if SAVE_FIGS:
#    plt.savefig("fig_gsd.pdf")
#else:
#    plt.show()
#plt.close()

# plotting water retention curve for CODE
#df_wrc[df_wrc.code == CODE].plot(x="preshead", y="theta", style='-bo',
#                                 legend=False, xlabel="Pressure (kPa)", ylabel="Volumetric water content (-)")
#if SAVE_FIGS:
#    plt.savefig("fig_wrc.pdf")
#else:
#    plt.show()
#plt.close()

# plotting all gsd curves
df_gsd.plot(x="particle_size", y="particle_fraction", logx=True, style='bo', alpha=0.1,
                                 legend=False, xlabel="Particle size (mm)", ylabel="Fraction (-)")
if SAVE_FIGS:
    plt.savefig("fig_gsd_all.pdf")
else:
    plt.show()
plt.close()

# plotting all water retention curves
df_wrc.plot(x="preshead", y="theta", logx=True, style='bo', alpha=0.1,
            legend=False, xlabel="Pressure (kPa)", ylabel="Volumetric water content (-)")
if SAVE_FIGS:
    plt.savefig("fig_wrc_all.pdf")
else:
    plt.show()
plt.close()

df_gen = pd.read_csv(SUBDIR + "general.csv")
#print("df_gen.count()")
#print(df_gen.count())
df_sand = df_gen[df_gen["texture"].isin(["sand"])]
#df_sand = df_gen[df_gen["texture"].astype(str).str.contains("sand")]
print("df_sand.count()")
print(df_sand.count())
#print(df_sand.texture)

# retrieve list of codes for a given soil texture
codes = df_gen[df_gen["texture"].isin(["sand"])].code.to_numpy()
#print(codes)

# plotting water retention curves for sands
ax = df_wrc[df_wrc.code.isin(codes) == False].plot(x="preshead", y="theta", style='bo', logx=True, alpha=0.1,
            legend=False, xlabel="Pressure (kPa)", ylabel="Volumetric water content (-)")
df_wrc[df_wrc.code.isin(codes)].plot(ax=ax,x="preshead", y="theta", style='ro', logx=True, alpha=0.1,
            legend=False, xlabel="Pressure (kPa)", ylabel="Volumetric water content (-)")
if SAVE_FIGS:
    plt.savefig("fig_wrc_select.pdf")
else:
    plt.show()
plt.close()

# plotting GSD for sands
ax = df_gsd[df_gsd.code.isin(codes) == False].plot(x="particle_size", y="particle_fraction", logx=True, style='bo', alpha=0.1,
            legend=False, xlabel="Particle size (mm)", ylabel="Fraction (-)")
ax = df_gsd[df_gsd.code.isin(codes)].plot(ax=ax, x="particle_size", y="particle_fraction", logx=True, style='ro', alpha=0.1,
            legend=False, xlabel="Particle size (mm)", ylabel="Fraction (-)")
if SAVE_FIGS:
    plt.savefig("fig_gsd_select.pdf")
else:
    plt.show()
plt.close()


# Selecting values for GSD curves (cumulative percentages for 7 values)
gsd_names = ['P2', 'P50', 'P100', 'P250', 'P500', 'P1000', 'P2000']
gsd_points = pd.Series([0.002, 0.05, 0.1, 0.25, 0.5, 1, 2])

#codes = [1110] #codes[0:2]

# counting the number of lines
selec_len = df_wrc[df_wrc.code.isin(codes)].preshead.count()
print('selec_len = ' + selec_len.astype(str))
#selec_len2 = df_wrc[df_wrc.code.isin(codes)].theta.count()
#print('selec_len2 = ' + selec_len2.astype(str))

df_select = pd.DataFrame(
    index=np.arange(selec_len), columns=['code'] + gsd_names + ['rho', 'suction', 'theta']
)
#print(df_select)
#print(df_select.describe())

icount = 0
for icode in codes:
#    print(icode)
    tmp1 = df_wrc.loc[df_wrc.code.isin([icode]),['preshead', 'theta']]
    if tmp1.count()['preshead'] != tmp1.count()['theta']:
        sys.exit("Length mismatch in wrc")
    tmp2 = df_gsd.loc[(df_gsd.code.isin([icode])),['particle_size', 'particle_fraction']]
    if tmp2.count()['particle_size'] != tmp2.count()['particle_fraction']:
        sys.exit("Length mismatch in gsd")
    if (tmp1.count()['preshead'] > 0) and (tmp2.particle_size.count() > 0):
        if (tmp2.particle_size.size != 7) or ((tmp2.particle_size.size == 7) and ((tmp2.particle_size.values != gsd_points.values).any())):
#            print("correction to gsd of: " + str(icode))
            f = interpolate.interp1d(tmp2.particle_size,tmp2.particle_fraction,
                                        bounds_error=False,fill_value=(0.,1.))
            tmp2 = pd.DataFrame(
                {
                'particle_size': gsd_points.values,
                'particle_fraction': f(gsd_points),
                }
            )        
        for iwrc in range(tmp1.count()['preshead']): #count() better than len() to exclude wrc with NaN values
            df_select.at[icount,'code'] = icode
            for ipoints in range(len(gsd_names)):
                tmp = tmp2[tmp2.particle_size == gsd_points[ipoints]].particle_fraction
                if len(tmp) > 0:
                    df_select.at[icount,gsd_names[ipoints]] = tmp.values[0]
            tmp = df_soil[df_soil.code.isin([icode])].bulk_density
            if len(tmp) > 0:
                df_select.at[icount,'rho'] = tmp.values[0]
            if len([tmp1.preshead.values[iwrc]]) > 0:
                df_select.at[icount,'suction'] = tmp1.preshead.values[iwrc]
            if len([tmp1.theta.values[iwrc]]) > 0:
                df_select.at[icount,'theta'] = tmp1.theta.values[iwrc]
            icount += 1

print(df_select)
print(df_select.describe())

#df_wrc.plot(x="preshead", y="theta", logx=True, style='bo', alpha=0.1,
#            legend=False, xlabel="Pressure (kPa)", ylabel="Volumetric water content (-)")
(df_select[gsd_names].T).plot(legend=False)
if SAVE_FIGS:
    plt.savefig("fig_gsd_select_corr.pdf")
else:
    plt.show()
plt.close()


#df_select = df_select[df_select['theta']<0.7]
#df_select = df_select[df_select['suction']<10000]
df_select = df_select[(df_select['theta']<0.7) & (df_select['suction']<1000)]
df_select.plot.scatter(x='suction', y='theta',legend=False)
if SAVE_FIGS:
    plt.savefig("fig_wrc_select_corr.pdf")
else:
    plt.show()
plt.close()

#outlier
#dataset[dataset['theta']>0.7]
#dataset[dataset['code']==1460]

df_select.to_csv(r'data_clean0.csv', index=False, header=True)
df_select.dropna(how="any").to_csv(r'data_clean.csv', index=False, header=True)
print('Data exported to csv files')

# on a enlevé les sols si pas de wrc ou pas de gsd

#format (pour chaque valeur de suction)
# code(en index?) GSDx7 bulk suctionx1 || thetax1

# TODO
# récupérer d'autres rho ?
