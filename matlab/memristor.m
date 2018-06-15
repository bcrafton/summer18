
dR = 1e6 - 5e4;
Qd = 1 / 1e6;

syms M(t) V;
cond = M(0) == 2e5;
ode = diff(M) == (dR / Qd) * (V / M);
sol(t, V) = dsolve(ode, cond);

t_in = linspace(0, 0.1, 1000);
v_in = linspace(0, 1, 1000);
r_out = zeros(1000, 1);

for i = 1:500
    r = sol( t_in(i), -1 * v_in(i) );
    r_out(i) = r;
end

for i = 1:500
    r = sol( t_in(i), v_in(i) );
    r_out(i) = r;
end

plot(v_in, r_out, 'b');
