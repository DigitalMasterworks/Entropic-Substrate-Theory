from astropy.io import fits

def load_nsa_positions(fits_path="nsa_v1_0_1.fits", id_col="NSAID", ra_col="RACAT", dec_col="DECCAT", z_col="Z"):
    """
    Load NSA FITS catalog and return a dict: {gal_id: (ra_deg, dec_deg, z)}
    Only reads columns you need to keep memory minimal.
    """
    print(f"Loading NSA positions from {fits_path} …")
    with fits.open(fits_path, memmap=True) as hdul:
        # Usually the table is in the first extension (HDU 1)
        data = hdul[1].data
        # Check the columns exist:
        cols = data.columns.names
        for c in (id_col, ra_col, dec_col, z_col):
            if c not in cols:
                raise KeyError(f"Column {c} not in NSA FITS (available: {cols[:5]})")
        ids   = data[id_col]
        ras   = data[ra_col]
        decs  = data[dec_col]
        zs    = data[z_col]
        # Build the dict
        pos = {}
        for i in range(len(ids)):
            gid = int(ids[i])
            pos[gid] = (float(ras[i]), float(decs[i]), float(zs[i]))
    print(f"Loaded {len(pos)} NSA positions.")
    return pos

if __name__ == "__main__":
    # Demo: just print the first few entries!
    pos = load_nsa_positions("nsa_v1_0_1.fits")
    print(list(pos.items())[:5])