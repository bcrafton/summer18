

addpath('./HspiceToolbox/');
addpath('./PolyfitnTools/');
colordef none;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = loadsig('indiveri_power.tr0');

% lssig(data)
power = evalsig(data, 'p_xneuron1');
t = evalsig(data, 'TIME');
% plot(t, power);

%disp(length(t));
%disp(length(power));

steps = length(t);
tp = 0;
for i = 1:steps-1
    dt = t(i+1) - t(i);
    p = power(i);
    tp = tp + dt * p;
end
%tp

steps = length(t);
tp = 0;
for i = 2:steps
    dt = t(i) - t(i-1);
    p = power(i);
    tp = tp + dt * p;
end
%tp

trapz(t, power)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = loadsig('indiveri_power1.tr0');

% lssig(data)
power = evalsig(data, 'p_xneuron1');
t = evalsig(data, 'TIME');
% plot(t, power);

steps = length(t);
tp = 0;
for i = 1:steps-1
    dt = t(i+1) - t(i);
    p = power(i);
    tp = tp + dt * p;
end
%tp 

steps = length(t);
tp = 0;
for i = 2:steps
    dt = t(i) - t(i-1);
    p = power(i);
    tp = tp + dt * p;
end
%tp

trapz(t, power)