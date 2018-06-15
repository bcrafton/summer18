function memristor 
    ron = 100;
    roff = 16e3;
    r_init = 11e3;
    dr = roff - ron;
    tmin = 0;
    tmax = 3;
    N = 500;
    OsaT = tmin : (tmax - tmin) / N : tmax;
    x0 = (roff-r_init)/dr;
    [t, x] = ode23t(@ode_memri_a, OsaT, x0);
    v = 1 * sin(2*pi*t);
    i = v ./ (roff-x*dr);
    
    flux = (t(2) - t(1)) * filter (1, [1 -1], v);
    charge = (t(2) - t(1)) * filter (1, [1 -1], i);
    
function dx=ode_memri_a(t, x)
    D = 10e-9;
    ron = 100;
    roff = 16e3;
    dr = roff - ron;
    uv = 1e-14;
    k=uv*ron/D^2;
    p=10;
    v = 1*sin(2*pi*t);
    fx = (1-(2*x-1) ^ (2*pi));
    dx = k*(v/(roff-x*dr))*fx;
        