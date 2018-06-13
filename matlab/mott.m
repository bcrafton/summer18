m1 = (1e5 - 5e4) / (0.9 - 0.0);
m2 = (1e6 - 1e5) / (1.0 - 0.9);
m3 = (8e5 - 1e6) / (0.1 - 1.0);
m4 = (5e4 - 8e5) / (0.0 - 0.1);

function [r, v, state] = memristor(r, v, state)

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