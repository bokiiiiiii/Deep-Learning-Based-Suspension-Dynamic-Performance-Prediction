function [MAX, RMS] = Two_Axle_Simulation_ISO8608(ms, mus, iy, cs, ks, kt, h, kratio, cratio, wr, cg, wb, alpha)
    
    % Input Parameters into variables
    mus1 = mus;
    mus2 = mus;
    ks1 = ks * kratio;
    ks2 = ks * (2 - kratio);
    cs1 = cs * cratio;
    cs2 = cs * (2 - cratio);
    kt1 = kt;
    kt2 = kt;

    % Distance to C.G
    a = wb / 2 + (cg - 50) / 100 * wb; 
    b = -wb / 2 + (cg - 50) / 100 * wb; 

    % Vehicle Velocity
    v = 30 * 1000 / 3600; % m/s
    
    % Road Parameters
    road_length = 100; 
    road_time = round(road_length / v, 4);

    N = 20000; 
    B = road_length / N; 
    s = 0:B:road_length - B; 

    load('class_DE_100m.mat'); 
    time_gen = s / v;
    inc = time_gen(1, 2);
    tw = round(wr / inc / v);

    % State Matrix
    g = 9.81;

    Abal = [
        0, 1, 0, 0, 0, 0, 0, 0;
        -(ks1 + ks2) / ms, -(cs1 + cs2) / ms, (a * ks1 + b * ks2) / ms, (a * cs1 + b * cs2) / ms, ks1 / ms, cs1 / ms, ks2 / ms, cs2 / ms;
        0, 0, 0, 1, 0, 0, 0, 0;
        (a * ks1 + b * ks2) / iy, (a * cs1 + b * cs2) / iy, -(a^2 * ks1 + b^2 * ks2) / iy, -(a^2 * cs1 + b^2 * cs2) / iy, -a * ks1 / iy, -a * cs1 / iy, -b * ks2 / iy, -b * cs2 / iy;
        0, 0, 0, 0, 0, 1, 0, 0;
        ks1 / mus1, cs1 / mus1, -a * ks1 / mus1, -a * cs1 / mus1, -(kt1 + ks1) / mus1, -cs1 / mus1, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 1;
        ks2 / mus2, cs2 / mus2, -b * ks2 / mus2, -b * cs2 / mus2, 0, 0, -(kt2 + ks2) / mus2, -cs2 / mus2
    ];

    Fbal = [0;
            -g * cos(alpha);
            0;
            -(ms + mus1 + mus2) * g * h * sin(alpha) / iy;
            0;
            -g * cos(alpha);
            0;
            -g * cos(alpha)];

    xbal = Abal \ -Fbal;

    % Settings for Axle Inputs
    delay2 = round((a - b) / v, 4); 
    add_time_1 = road_time + inc * (1:round(delay2 / inc)); 
    time = [time_gen, add_time_1];
    add_zero_1 = zeros(1, length(add_time_1));
    zr1 = [zr, add_zero_1];
    zr2 = [add_zero_1, zr];
    u = [zr1', zr2'];

    % Initialize Results
    xResult = zeros(length(time_gen) - tw - 10, 8);
    dxResult = zeros(length(time_gen) - tw - 10, 1);
    dthResult = zeros(length(time_gen) - tw - 10, 1);

    % Run Simulation
    x0 = xbal;
    for index = 10 + tw:length(u) - tw - 10
        road = u(index, :);
        deltax = car(a, b, ms, iy, ks1, cs1, mus1, kt1, ks2, cs2, mus2, kt2, x0, road, alpha, h);
        dxResult(index - (10 + tw) + 1) = deltax(2); 
        dthResult(index - (10 + tw) + 1) = deltax(4);
        xResult(index - (10 + tw) + 1, :) = (x0 + deltax * inc)' - xbal';
        x0 = x0 + deltax * inc;
    end

    % Post-Processing
    x = xResult;
    sws1 = abs(x(:, 1) - a * x(:, 3) - x(:, 5));
    sws2 = abs(x(:, 1) - b * x(:, 3) - x(:, 7));
    dz1 = x(:, 5) - u(10 + tw:length(u) - tw - 10, 1);
    dz2 = x(:, 7) - u(10 + tw:length(u) - tw - 10, 2);
    dtl1 = -kt1 * dz1; 
    dtl2 = -kt2 * dz2;

    calculated1 = obs_point(x(:, 1), x(:, 3), dxResult, dthResult, x(:, 4), a);
    zs1 = calculated1(:, 1);
    zsdotdot1 = calculated1(:, 2);
    calculated2 = obs_point(x(:, 1), x(:, 3), dxResult, dthResult, x(:, 4), b);
    zs2 = calculated2(:, 1);
    zsdotdot2 = calculated2(:, 2);

    % Calculate MAX and RMS values
    MAX = [max(x(:, 1)), max(dxResult), max(dtl1), max(sws1), max(x(:, 3)), max(abs(dthResult)), max(dtl2), max(sws2)];

    RMS = [rms(x(:, 1)), rms(dxResult), rms(dtl1), rms(sws1), rms(x(:, 3)), rms(dthResult), rms(dtl2), rms(sws2)];
end


% Establishing State space model
function dx = car(a, b, ms, iy, ks1, cs1, mus1, kt1, ks2, cs2, mus2, kt2, x, road, alpha, h)
    g = 9.81;

    A = [
        0, 1, 0, 0, 0, 0, 0, 0;
        -(ks1 + ks2) / ms, -(cs1 + cs2) / ms, (a * ks1 + b * ks2) / ms, (a * cs1 + b * cs2) / ms, ks1 / ms, cs1 / ms, ks2 / ms, cs2 / ms;
        0, 0, 0, 1, 0, 0, 0, 0;
        (a * ks1 + b * ks2) / iy, (a * cs1 + b * cs2) / iy, -(a^2 * ks1 + b^2 * ks2) / iy, -(a^2 * cs1 + b^2 * cs2) / iy, -a * ks1 / iy, -a * cs1 / iy, -b * ks2 / iy, -b * cs2 / iy;
        0, 0, 0, 0, 0, 1, 0, 0;
        ks1 / mus1, cs1 / mus1, -a * ks1 / mus1, -a * cs1 / mus1, -(kt1 + ks1) / mus1, -cs1 / mus1, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 1;
        ks2 / mus2, cs2 / mus2, -b * ks2 / mus2, -b * cs2 / mus2, 0, 0, -(kt2 + ks2) / mus2, -cs2 / mus2
    ];

    B = [0, 0;
         0, 0;
         0, 0;
         0, 0;
         0, 0;
         (kt1 / mus1), 0;
         0, 0;
         0, (kt2 / mus2)];

    F = [0;
         -g * cos(alpha);
         0;
         -(ms + mus1 + mus2) * g * h * sin(alpha) / iy;
         0;
         -g * cos(alpha);
         0;
         -g * cos(alpha)];

    dx = A * x + B * road' + F;
end


% Calculate sprung mass and sprung mass acceleration at observation points
function calculated = obs_point(sprung_dis, pitch, sprung_acc, angular_acc, angular_vel, distance)
    obs_dis = sprung_dis + (-distance * tan(pitch));
    obs_acc = sprung_acc + (angular_acc .* -distance) - ((angular_vel .^ 2) .* -distance);
    calculated = [obs_dis, obs_acc];
end
