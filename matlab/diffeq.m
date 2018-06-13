
C1 = 7e-9;
C2 = 2e-9;
%RS1 = 5e4;
%RS2 = 5e4;
R1 = 1e5; % 100k
R2 = 1e5; % 100k
Vdc = -0.9;
Vdc2 = 0.9;
RL = 10e9;

syms V(t) V2(t) I RS1 RS2;
condV = V(0) == 0;
condV2 = V2(0) == 0;

ode1 = C1 * diff(V) == I - (1/RS1) * (V - Vdc) - (1/R2) * (V - V2);
ode2 = C2 * diff(V2) == (1/R2) * (V - V2) - (1/RS2) * (V2 - Vdc2) - (1/RL) * V2;
[solV(t, I, RS1, RS2), solV2(t, I, RS1, RS2)] = dsolve([ode1; ode2], [condV; condV2]);

t_in = linspace(0, 1, 1000);
I_in = linspace(1e-6, 1e-6, 1000);

y1 = zeros(1000, 1);
y2 = zeros(1000, 1);
r1 = zeros(1000, 1);
r2 = zeros(1000, 1);

RS1_in = 5e4;
RS2_in = 5e4;
d1 = 0;
d2 = 0;
state_RS1 = 0;
state_RS2 = 0;

for i = 1:1000
    y1(i) = solV(t_in(i), I_in(i), RS1_in, RS2_in);
    y2(i) = solV2(t_in(i), I_in(i), RS1_in, RS2_in);
    r1(i) = RS1_in;
    r2(i) = RS2_in;
    [RS1_in, d1, state_RS1] = memristor(RS1_in, y1(i) - Vdc, state_RS1);
    [RS2_in, d2, state_RS2] = memristor(RS2_in, y2(i) - Vdc2, state_RS2);
end

plot(t_in, r1, t_in, r2);
% plot(t_in, y1, 'b', t_in, y2, 'r');

function [r, v, state] = memristor(r, v, state)

    m1 = (1e5 - 5e4) / (0.9 - 0.0);
    m2 = (1e6 - 1e5) / (1.0 - 0.9);
    m3 = (8e5 - 1e6) / (0.1 - 1.0);
    m4 = (5e4 - 8e5) / (0.0 - 0.1);

    if (v <= 0.9 && state == 0)
        r = m1 * v + 5e4;
    elseif (v > 0.9 && state == 0)
        r = m2 * (v - 0.9) + 1e5;
        if (r >= 1e6)
            state = 1;
        end

    elseif (v >= 0.1 && state == 1)
        r = 1e6 - m3 * (0.9 - v);
    elseif (v < 0.1 && state == 1)
        r = 8e5 - m4 * (0.1 - v);
        if (r <= 5e4)
            state = 0;
        end

    else
        fprintf ("this should never happen %f %f %f\n", r, v, state);
    end
    
    r = max(r, 5e4);
    r = min(r, 1e6);
    % r = clip(r, 5e4, 1e6);
end
