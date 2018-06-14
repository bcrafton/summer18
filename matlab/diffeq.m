
C1 = 3e-9;
C2 = 2e-9;
R1 = 1e5; % 100k
R2 = 1e5; % 100k
Vdc = -0.9;
Vdc2 = 0.9;
RL = 1e9;

syms V(t) V2(t) I RS1 RS2;
condV = V(0) == 0;
condV2 = V2(0) == 0;

ode1 = C1 * diff(V) == I - (1/RS1) * (V - Vdc) - (1/R2) * (V - V2);
ode2 = C2 * diff(V2) == (1/R2) * (V - V2) - (1/RS2) * (V2 - Vdc2) - (1/RL) * V2;
[solV(t, I, RS1, RS2), solV2(t, I, RS1, RS2)] = dsolve([ode1; ode2], [condV; condV2]);

t_in = linspace(0, 0.01, 1000);
I_in = [linspace(0, 0, 500), linspace(0, 1e-6, 100), linspace(1e-6, 1e-6, 500)];

y1 = zeros(1000, 1);
y2 = zeros(1000, 1);
r1 = zeros(1000, 1);
r2 = zeros(1000, 1);
vr1 = zeros(1000, 1);
vr2 = zeros(1000, 1);

RS1_in = 1e6;
RS2_in = 1e6;
state_RS1 = 0;
state_RS2 = 0;

for i = 1:1000
    y1(i) = solV(t_in(i), I_in(i), RS1_in, RS2_in);
    y2(i) = solV2(t_in(i), I_in(i), RS1_in, RS2_in);
    r1(i) = RS1_in;
    r2(i) = RS2_in;
    vr1(i) = y1(i) - Vdc;
    vr2(i) = Vdc2 - y2(i);
    [RS1_in, state_RS1] = memristor(y1(i) - Vdc, state_RS1);
    [RS2_in, state_RS2] = memristor(Vdc2 - y2(i), state_RS2);
end

% plot(t_in, r1, t_in, r2);
% plot(t_in, y1, 'b', t_in, y2, 'r');
% plot(t_in, y2, 'r');
plot(t_in, vr1, 'b', t_in, vr2, 'r');

function [r, state] = memristor(v, state)

    m1 = (1e6 - 8e5) / (0.95 - 0.0);
    m2 = (8e5 - 5e4) / (1.0 - 0.95);
    m3 = (5e4 - 5e5) / (0.55 - 1.0);
    m4 = (5e5 - 1e6) / (0.5 - 0.55);

    if (state == 0 && v <= 1.0)
        r = 1e6 - m1 * v;
        r = max(r, 8e5);
        r = min(r, 1e6);
        state = 0;
    elseif (state == 0 && v > 0.95)
        r = 8e5 - m2 * (v - 0.95);
        r = max(r, 5e4);
        r = min(r, 8e5);
        if (v > 1.0)
            state = 1;
        end
    elseif (state == 1 && v >= 0.55)
        r = 5e4 + m3 * (1.0 - v);
        r = max(r, 5e4);
        r = min(r, 5e5);
        state = 1;
    elseif (state == 1 && v < 0.55)
        r = 5e5 + m4 * (0.55 - v);
        r = max(r, 5e5);
        r = min(r, 1e6);
        if (v < 0.5)
            state = 0;
        end
    else
        disp("should never get here");
    end

end
