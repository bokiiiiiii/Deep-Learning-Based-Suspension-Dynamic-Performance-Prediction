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
for i = 0.3:0.02:1.7

    % Create random inputs
    ms      = ms_origin     * 1; % Random value between 0.5 and 1.5
    mus     = mus_origin    * 1;
    iy      = iy_origin     * 1;
    cs      = cs_origin     * 1;
    ks      = ks_origin     * i;
    kt      = kt_origin     * 1;
    h       = h_origin      * 1;
    wr      = wr_origin     * 1;
    cg      = cg_origin     * 1;
    wb      = wb_origin     * 1;
    alpha   = alpha_origin  * 1;
    kratio  = kratio_origin * 1;
    cratio  = cratio_origin * 1;

    % Run the simulation
    [MAX, RMS] = Two_Axle_Simulation_ISO8608(ms, mus, iy, cs, ks, kt, h, kratio, cratio, wr, cg, wb, alpha);
    data_row(end+1,:) = [ms, mus, iy, cs, ks, kt, h, kratio, cratio, wr, cg, wb, alpha, MAX, RMS];
    
end


%% Append the new data row
filename = 'Test.xlsx';
writematrix(data_row, filename, 'WriteMode', 'append');

