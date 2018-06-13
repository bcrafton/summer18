syms y(t);
ode = diff(y, t) == t*y;
ySol = dsolve(ode);

% disp (ySol);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C = 1e-9;
I = linspace(0, 1e-9, 10);
I = 1e-9;
R1 = 1e5; % 100k
R2 = 1e5; % 100k
RS1 = 10e6; % 1e6 / 20
RS2 = 10e6; % 1e6 / 20
Vdc = 0.9;
Vdc2 = -0.9;
RL = 10e9;

syms V(t) V2(t);
condV = V(0) == 0;
condV2 = V2(0) == 0;
ode1 = C * diff(V) == I - 1/RS1*(V - Vdc) - 1/R2*(V - V2);
ode2 = C * diff(V2) == 1/R2 * (V - V2) - 1/RS2 * (V2 - Vdc2) - (1/RL) * V2;
sol = dsolve([ode1; ode2], [condV; condV2]);

hold on;
fplot(sol.V);
fplot(sol.V2);
hold off;
