from drizzlepac import astrodrizzle


def main():
    in_frame  = 'r_n0060.fits'
    out_frame = 'drz_{}'.format(in_frame)
    wgt_frame = 'wgt_{}'.format(in_frame)
    log_file  = '{}.log'.format(in_frame)

    astrodrizzle.AstroDrizzle(in_frame, output='drz.fits')

if __name__=='__main__':
    main()
