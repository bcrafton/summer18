
C = 1e-9;
R1 = 1e5; % 100k
R2 = 1e5; % 100k
RS1 = 10e6; % 1e6 / 20
RS2 = 10e6; % 1e6 / 20
Vdc = 0.9;
Vdc2 = -0.9;
RL = 10e9;

%syms V(t) V2(t) I RS1 RS2;
syms V(t) V2(t) I;
condV = V(0) == 0;
condV2 = V2(0) == 0;

ode1 = C * diff(V) == I - (1/RS1) * (V - Vdc) - (1/R2) * (V - V2);
ode2 = C * diff(V2) == (1/R2) * (V - V2) - (1/RS2) * (V2 - Vdc2) - (1/RL) * V2;

[solV(t, I), solV2(t, I)] = dsolve([ode1; ode2], [condV; condV2]);

t_in = linspace(0, 1, 1000);
I_in = linspace(1e-6, 1e-6, 1000);

y1 = zeros(1000, 1);
y2 = zeros(1000, 1);

for i = 1:1000
    y1(i) = solV(t_in(i), I_in(i));
    y2(i) = solV2(t_in(i), I_in(i));
    
end

plot(t_in, y1, t_in, y2);

%hold on;
%fplot(sol.V);
%fplot(sol.V2);
%hold off;

function [r, v, state] = memristor(r, v, state)

    m1 = (1e5 - 5e4) / (0.9 - 0.0);
    m2 = (1e6 - 1e5) / (1.0 - 0.9);
    m3 = (8e5 - 1e6) / (0.1 - 1.0);
    m4 = (5e4 - 8e5) / (0.0 - 0.1);

    if (v >= 0.0 && v <= 0.9 && state == 0)
        r = m1 * v + 5e4;
    elseif (v > 0.9 && v <= 1.0 && state == 0)
        r = m2 * (v - 0.9) + 1e5;
        if (r >= 1e6)
            state = 1;
        end

    elseif (v >= 0.1 && v <= 1.0 && state == 1)
        r = 1e6 - m3 * (0.9 - v);
    elseif (v >= 0.0 && v < 0.1 && state == 1)
        r = 8e5 - m4 * (0.1 - v);
        if (r <= 5e4)
            state = 0;
        end

    else
        disp ("this should never happen")
    end
end
