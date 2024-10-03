function [MAX, RMS] = Three_Axle_Simulation_ISO8608(ms, mus, iy, cs, ks, kt, h, kratio, cratio, wr, cg, wb, alpha)

    % Input Parameters into variables
    mus1 = mus; % unsprung mass for axle 1
    mus2 = mus; % unsprung mass for axle 2
    mus3 = mus; % unsprung mass for axle 3
    ks1 = ks * kratio; % suspension stiffness for axle 1
    ks2 = ks * (3 - 2 * kratio); % suspension stiffness for axle 2
    ks3 = ks * kratio; % suspension stiffness for axle 3
    cs1 = cs * cratio; % damping coefficient for axle 1
    cs2 = cs * (3 - 2 * cratio); % damping coefficient for axle 2
    cs3 = cs * cratio; % damping coefficient for axle 3
    kt1 = kt; % tire stiffness for axle 1
    kt2 = kt; % tire stiffness for axle 2
    kt3 = kt; % tire stiffness for axle 3

    % Distance to C.G for each axle
    a = wb / 2 + (cg - 50) / 100 * wb; % distance of axle 1 from CG
    b = -wb / 2 + (cg - 50) / 100 * wb; % distance of axle 2 from CG
    c = -wb / 2 + (cg - 50) / 100 * wb - 2.425; % distance of axle 3 from CG

    % Vehicle velocity
    v = 30 * 1000/3600; % velocity in m/s

    % Road parameters
    road_length = 100; % length of the road (m)
    road_time = round(road_length / v, 4); % total time for simulation based on road length

    % Number of samples and spacing
    N = 20000; % number of data points
    B = road_length / N; % sample interval (m)
    s = 0:B:road_length - B; % road surface profile along the distance

    % Load ISO road profile data
    load('class_DE_100m.mat'); % replace with your road profile data file
    time_gen = s / v;
    inc = time_gen(1, 2); % time increment
    tw = round(wr / inc / v); % wheel radius total step

    % State Matrix
    g = 9.81; % acceleration due to gravity

    Abal = [
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
            - (ks1 + ks2 + ks3) / ms, - (cs1 + cs2 + cs3) / ms, (a * ks1 + b * ks2 + c * ks3) / ms, (a * cs1 + b * cs2 + c * cs3) / ms, ks1 / ms, cs1 / ms, ks2 / ms, cs2 / ms, ks3 / ms, cs3 / ms;
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
            (a * ks1 + b * ks2 + c * ks3) / iy, (a * cs1 + b * cs2 + c * cs3) / iy, - (a ^ 2 * ks1 + b ^ 2 * ks2 + c ^ 2 * ks3) / iy, - (a ^ 2 * cs1 + b ^ 2 * cs2 + c ^ 2 * cs3) / iy, -a * ks1 / iy, -a * cs1 / iy, -b * ks2 / iy, -b * cs2 / iy, -c * ks3 / iy, -c * cs3 / iy;
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
            ks1 / mus1, cs1 / mus1, -a * ks1 / mus1, -a * cs1 / mus1, - (kt1 + ks1) / mus1, -cs1 / mus1, 0, 0, 0, 0;
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
            ks2 / mus2, cs2 / mus2, -b * ks2 / mus2, -b * cs2 / mus2, 0, 0, - (kt2 + ks2) / mus2, -cs2 / mus2, 0, 0;
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
            ks3 / mus3, cs3 / mus3, -c * ks3 / mus3, -c * cs3 / mus3, 0, 0, 0, 0, - (kt3 + ks3) / mus3, -cs3 / mus3
            ];

    Fbal = [0;
            -g * cos(alpha);
            0;
            - (ms + mus1 + mus2 + mus3) * g * h * sin(alpha) / iy;
            0;
            -g * cos(alpha);
            0;
            -g * cos(alpha);
            0;
            -g * cos(alpha)];

    xbal = Abal \ -Fbal;

    % Settings for Axle Inputs
    delay2 = round((a - b) / v, 4); % time delay between axle 1 and 2
    delay3 = round((a - c) / v, 4); % time delay between axle 1 and 3
    add_time_1 = road_time + inc * (1:round(delay3 / inc));
    time = [time_gen, add_time_1];
    add_zero_1 = zeros(1, length(add_time_1));
    zr1 = [zr, add_zero_1];
    zr2 = [add_zero_1, zr];
    zr3 = [add_zero_1, zr];
    u = [zr1', zr2', zr3'];

    % Initialize Results
    xResult = zeros(length(time_gen) - tw - 10, 10); % updated for 10 state variables
    dxResult = zeros(length(time_gen) - tw - 10, 1);
    dthResult = zeros(length(time_gen) - tw - 10, 1);

    % Run Simulation
    x0 = xbal;

    for index = 10 + tw:length(u) - tw - 10
        road = u(index, :);
        deltax = car(a, b, c, ms, iy, ks1, cs1, mus1, kt1, ks2, cs2, mus2, kt2, ks3, cs3, mus3, kt3, x0, road, alpha, h);
        dxResult(index - (10 + tw) + 1) = deltax(2); % acceleration
        dthResult(index - (10 + tw) + 1) = deltax(4); % angular acceleration
        xResult(index - (10 + tw) + 1, :) = (x0 + deltax * inc)' - xbal';
        x0 = x0 + deltax * inc;
    end

    % Post-Processing
    x = xResult;
    sws1 = abs(x(:, 1) - a * x(:, 3) - x(:, 5)); % suspension work for axle 1
    sws2 = abs(x(:, 1) - b * x(:, 3) - x(:, 7)); % suspension work for axle 2
    sws3 = abs(x(:, 1) - c * x(:, 3) - x(:, 9)); % suspension work for axle 3
    dz1 = x(:, 5) - u(10 + tw:length(u) - tw - 10, 1); % tire displacement axle 1
    dz2 = x(:, 7) - u(10 + tw:length(u) - tw - 10, 2); % tire displacement axle 2
    dz3 = x(:, 9) - u(10 + tw:length(u) - tw - 10, 3); % tire displacement axle 3
    dtl1 = -kt1 * dz1; % dynamic tire load axle 1
    dtl2 = -kt2 * dz2; % dynamic tire load axle 2
    dtl3 = -kt3 * dz3; % dynamic tire load axle 3

    calculated1 = obs_point(x(:, 1), x(:, 3), dxResult, dthResult, x(:, 4), a); % observation point axle 1
    zs1 = calculated1(:, 1);
    zsdotdot1 = calculated1(:, 2);
    calculated2 = obs_point(x(:, 1), x(:, 3), dxResult, dthResult, x(:, 4), b); % observation point axle 2
    zs2 = calculated2(:, 1);
    zsdotdot2 = calculated2(:, 2);
    calculated3 = obs_point(x(:, 1), x(:, 3), dxResult, dthResult, x(:, 4), c); % observation point axle 3
    zs3 = calculated3(:, 1);
    zsdotdot3 = calculated3(:, 2);

    % Calculate MAX and RMS values
    MAX = [max(x(:, 1)), max(dxResult), max(dtl1), max(sws1), max(x(:, 3)), max(abs(dthResult)), max(dtl2), max(sws2), max(dtl3), max(sws3)];
    RMS = [rms(x(:, 1)), rms(dxResult), rms(dtl1), rms(sws1), rms(x(:, 3)), rms(dthResult), rms(dtl2), rms(sws2), rms(dtl3), rms(sws3)];

end

function dx = car(a, b, c, ms, iy, ks1, cs1, mus1, kt1, ks2, cs2, mus2, kt2, ks3, cs3, mus3, kt3, x, road, alpha, h)
    g = 9.81;

    A = [
         0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
         - (ks1 + ks2 + ks3) / ms, - (cs1 + cs2 + cs3) / ms, (a * ks1 + b * ks2 + c * ks3) / ms, (a * cs1 + b * cs2 + c * cs3) / ms, ks1 / ms, cs1 / ms, ks2 / ms, cs2 / ms, ks3 / ms, cs3 / ms;
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
         (a * ks1 + b * ks2 + c * ks3) / iy, (a * cs1 + b * cs2 + c * cs3) / iy, - (a ^ 2 * ks1 + b ^ 2 * ks2 + c ^ 2 * ks3) / iy, - (a ^ 2 * cs1 + b ^ 2 * cs2 + c ^ 2 * cs3) / iy, -a * ks1 / iy, -a * cs1 / iy, -b * ks2 / iy, -b * cs2 / iy, -c * ks3 / iy, -c * cs3 / iy;
         0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
         ks1 / mus1, cs1 / mus1, -a * ks1 / mus1, -a * cs1 / mus1, - (kt1 + ks1) / mus1, -cs1 / mus1, 0, 0, 0, 0;
         0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
         ks2 / mus2, cs2 / mus2, -b * ks2 / mus2, -b * cs2 / mus2, 0, 0, - (kt2 + ks2) / mus2, -cs2 / mus2, 0, 0;
         0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
         ks3 / mus3, cs3 / mus3, -c * ks3 / mus3, -c * cs3 / mus3, 0, 0, 0, 0, - (kt3 + ks3) / mus3, -cs3 / mus3
         ];

    B = [0, 0, 0;
         0, 0, 0;
         0, 0, 0;
         0, 0, 0;
         0, 0, 0;
         (kt1 / mus1), 0, 0;
         0, 0, 0;
         0, (kt2 / mus2), 0;
         0, 0, 0;
         0, 0, (kt3 / mus3)];

    F = [0;
         -g * cos(alpha);
         0;
         - (ms + mus1 + mus2 + mus3) * g * h * sin(alpha) / iy;
         0;
         -g * cos(alpha);
         0;
         -g * cos(alpha);
         0;
         -g * cos(alpha)];

    dx = A * x + B * road' + F;
end

function calculated = obs_point(sprung_dis, pitch, sprung_acc, angular_acc, angular_vel, distance)
    obs_dis = sprung_dis + (-distance * tan(pitch)); % displacement at observation point
    obs_acc = sprung_acc + (angular_acc .* -distance) - ((angular_vel .^ 2) .* -distance); % acceleration at observation point
    calculated = [obs_dis, obs_acc];
end
