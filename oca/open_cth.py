import xarray as xr

path_to_data = "/data/sat/mtg/fci/oca/2025/05/22"

filename = 'W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-2-OCA--FD------NC4E_C_EUMT_20250522234518_L2PF_OPE_20250522233000_20250522234000_N__C_0142_0000.nc' 

ds = xr.open_dataset(f"{path_to_data}/{filename}" , engine='h5netcdf')   

print(ds)

#print var names
print(ds['mtg_geos_projection'].variables)