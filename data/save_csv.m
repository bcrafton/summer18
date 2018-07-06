

addpath('../HspiceToolbox/');
addpath('../PolyfitnTools/');
colordef none;

leak = loadsig('leak.tr0');
inv_fb = loadsig('inv_fb.tr0');
reset = loadsig('reset.tr0');
inv_slew = loadsig('inv_slew.tr0');

%%%%%%%%%%%%%%%%%%%%%%
% leak
v_vmem = evalsig(leak, 'v_vmem');
i_m20 = evalsig(leak, 'i_m20');

csvwrite('leak_vmem.csv', v_vmem);
csvwrite('leak_m20.csv', i_m20);

%%%%%%%%%%%%%%%%%%%%%%
% src_flw, inv_fb
v_vmem = evalsig(inv_fb, 'v_vmem');
v_vo1 = evalsig(inv_fb, 'v_vo1');
i_m7 = evalsig(inv_fb, 'i_m7');

csvwrite('fb_vmem.csv', v_vmem);
csvwrite('fb_vo1.csv', v_vo1);
csvwrite('fb_m7.csv', i_m7);
%%%%%%%%%%%%%%%%%%%%%%
% reset
v_vmem = evalsig(reset, 'v_vmem');
v_vo2 = evalsig(reset, 'v_vo2');
i_m12 = evalsig(reset, 'i_m12');

csvwrite('rst_vmem.csv', v_vmem);
csvwrite('rst_vo2.csv', v_vo2);
csvwrite('rst_m12.csv', i_m12);
%%%%%%%%%%%%%%%%%%%%%%
% inv_slew
v_vo1 = evalsig(inv_slew, 'v_vo1');
v_vo2 = evalsig(inv_slew, 'v_vo2');
i_vso2 = evalsig(inv_slew, 'i_vso2');

csvwrite('slew_vo1.csv', v_vo1);
csvwrite('slew_vo2.csv', v_vo2);
csvwrite('slew_co2.csv', i_vso2);
%%%%%%%%%%%%%%%%%%%%%%



