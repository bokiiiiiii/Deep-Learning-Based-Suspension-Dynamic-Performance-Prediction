%% Initialize
clear
clc
close all

%% Set up vehicle parameters as simulation inputs
ms_origin      = 20337.8/2*2/4;        % sprung mass 
mus_origin     = 458.4;                % unsprung mass
iy_origin      = 562239.5791/2*2/4;    % pitch moment of inertia
cs_origin      = 11522.5191099067;     % damping coefficient
ks_origin      = 128710;               % spring stiffness
kt_origin      = 840857;               % tire stiffness
h_origin       = 1.15;                 % c.g. height (m)
wr_origin      = 1.189/2;              % wheel radius (m)
cg_origin      = 50;                   % c.g. position (%)
wb_origin      = 4.85;                 % wheelbase (m)
alpha_origin   = 0;                    % slope
kratio_origin  = 1;                    % spring stiffness ratio
cratio_origin  = 1;                    % damping coefficient ratio

%% Get random simulation data
data_row = [];
[MAX_origin, RMS_origin] = Two_Axle_Simulation_ISO8608(ms_origin, mus_origin, iy_origin, cs_origin, ks_origin, ...
    kt_origin, h_origin, kratio_origin, cratio_origin, wr_origin, cg_origin, wb_origin, alpha_origin);
for i = 0.2:0.02:1.8

    % Create random inputs
    ms      = ms_origin*i; % Random value between 0.5 and 1.5
    mus     = mus_origin;
    iy      = iy_origin;
    cs      = cs_origin;
    ks      = ks_origin;
    kt      = kt_origin;
    h       = h_origin;
    wr      = wr_origin;
    cg      = cg_origin;
    wb      = wb_origin;
    alpha   = alpha_origin;
    kratio  = kratio_origin;
    cratio  = cratio_origin;

    % Run the simulation
    [MAX, RMS] = Two_Axle_Simulation_ISO8608(ms, mus, iy, cs, ks, kt, h, kratio, cratio, wr, cg, wb, alpha);
    SDPI = 0.33 * (0.6 * RMS(2) / RMS_origin(2) + 0.2 * RMS(5) / RMS_origin(5) + 0.2 * RMS(6) / RMS_origin(6)) ...
        + 0.01 * (MAX(4) + MAX(8)) / (MAX_origin(4) + MAX_origin(8)) ...
        + 0.66 * (RMS(3) + RMS(7)) / (RMS_origin(3) + RMS_origin(7));
    data_row(end+1,:) = [ms, mus, iy, cs, ks, kt, h, kratio, cratio, wr, cg, wb, alpha, MAX, RMS, SDPI];
    
end


%% Append the new data row
filename = 'Test.xlsx';
writematrix(data_row, filename, 'WriteMode', 'overwrite');

