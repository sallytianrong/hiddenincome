clear all; close all; clc;

% Parameters
n = 100;

% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;
a1var = linspace(0.1,0.9,n);

% Pareto weight
alpha = 0.5;
alphavar = linspace(0.1,0.9,n);

% Time discounting
beta = 0.9;
betavar = linspace(0.8,0.97,n);

% Cost of hiding
delta = 0.8;
deltavar = linspace(0.1,1,n);

% income of agent 2
y2 = 3;
y2var = [linspace(1,1.5,n/2) linspace(1.5,5,n/2)];

%% simulation 1: vary preference of private good
MPP_1 = zeros(1,n);
MPC1_fb_1 = zeros(1,n);
MPC1_1 = zeros(1,n);
MPC2_fb_1 = zeros(1,n);
MPC2_1 = zeros(1,n);
MPCQ_fb_1 = zeros(1,n);
MPCQ_1 = zeros(1,n);
for i = 1:n
    [MPP_1(i), MPC1_fb_1(i), MPC1_1(i), MPC2_fb_1(i), MPC2_1(i), MPCQ_fb_1(i), MPCQ_1(i)] = infinitegame(a1var(i), a2, alpha, beta, delta, y2);
end

% graph
figure;
subplot(2,2,1);
plot(a1var, ones(1,n), a1var, MPP_1);
legend('no hiding','hiding');
xlabel('Preference of private good')
title('Marginal Propensity to Report');
subplot(2,2,2);
plot(a1var, MPC1_fb_1, a1var, MPC1_1);
legend('no hiding','hiding');
xlabel('Preference of private good')
title('MPC of C1');
subplot(2,2,3);
plot(a1var, MPC2_fb_1, a1var, MPC2_1);
legend('no hiding','hiding');
xlabel('Preference of private good')
title('MPC of C2');
subplot(2,2,4);
plot(a1var, MPCQ_fb_1, a1var, MPCQ_1);
xlabel('Preference of private good')
legend('no hiding','hiding');
title('MPC of Q');

%% simulation 2: vary Pareto weight
MPP_2 = zeros(1,n);
MPC1_fb_2 = zeros(1,n);
MPC1_2 = zeros(1,n);
MPC2_fb_2 = zeros(1,n);
MPC2_2 = zeros(1,n);
MPCQ_fb_2 = zeros(1,n);
MPCQ_2 = zeros(1,n);
for i = 1:n
    [MPP_2(i), MPC1_fb_2(i), MPC1_2(i), MPC2_fb_2(i), MPC2_2(i), MPCQ_fb_2(i), MPCQ_2(i)] = infinitegame(a1, a2, alphavar(i), beta, delta, y2);
end

% graph
figure;
subplot(2,2,1);
plot(alphavar, ones(1,n), alphavar, MPP_2);
legend('no hiding','hiding');
xlabel('Pareto weight of Agent 1')
title('Marginal Propensity to Report');
subplot(2,2,2);
plot(alphavar, MPC1_fb_2, alphavar, MPC1_2);
legend('no hiding','hiding');
xlabel('Pareto weight of Agent 1')
title('MPC of C1');
subplot(2,2,3);
plot(alphavar, MPC2_fb_2, alphavar, MPC2_2);
legend('no hiding','hiding');
xlabel('Pareto weight of Agent 1')
title('MPC of C2');
subplot(2,2,4);
plot(alphavar, MPCQ_fb_2, alphavar, MPCQ_2);
xlabel('Pareto weight of Agent 1')
legend('no hiding','hiding');
title('MPC of Q');

%% simulation 3: vary time discounting factor
MPP_3 = zeros(1,n);
MPC1_fb_3 = zeros(1,n);
MPC1_3 = zeros(1,n);
MPC2_fb_3 = zeros(1,n);
MPC2_3 = zeros(1,n);
MPCQ_fb_3 = zeros(1,n);
MPCQ_3 = zeros(1,n);
for i = 1:n
    [MPP_3(i), MPC1_fb_3(i), MPC1_3(i), MPC2_fb_3(i), MPC2_3(i), MPCQ_fb_3(i), MPCQ_3(i)] = infinitegame(a1, a2, alpha, betavar(i), delta, y2);
end

% graph
figure;
subplot(2,2,1);
plot(betavar, ones(1,n), betavar, MPP_3);
legend('no hiding','hiding');
xlabel('Time discounting factor')
title('Marginal Propensity to Report');
subplot(2,2,2);
plot(betavar, MPC1_fb_3, betavar, MPC1_3);
legend('no hiding','hiding');
xlabel('Time discounting factor')
title('MPC of C1');
subplot(2,2,3);
plot(betavar, MPC2_fb_3, betavar, MPC2_3);
legend('no hiding','hiding');
xlabel('Time discounting factor')
title('MPC of C2');
subplot(2,2,4);
plot(betavar, MPCQ_fb_3, betavar, MPCQ_3);
xlabel('Time discounting factor')
legend('no hiding','hiding');
title('MPC of Q');

%% simulation 4: vary cost of hiding
MPP_4 = zeros(1,n);
MPC1_fb_4 = zeros(1,n);
MPC1_4 = zeros(1,n);
MPC2_fb_4 = zeros(1,n);
MPC2_4 = zeros(1,n);
MPCQ_fb_4 = zeros(1,n);
MPCQ_4 = zeros(1,n);
for i = 1:n
    [MPP_4(i), MPC1_fb_4(i), MPC1_4(i), MPC2_fb_4(i), MPC2_4(i), MPCQ_fb_4(i), MPCQ_4(i)] = infinitegame(a1, a2, alpha, beta, deltavar(i), y2);
end

% graph
figure;
subplot(2,2,1);
plot(deltavar, ones(1,n), deltavar, MPP_4);
legend('no hiding','hiding');
xlabel('Inverse cost of hiding')
title('Marginal Propensity to Report');
subplot(2,2,2);
plot(deltavar, MPC1_fb_4, deltavar, MPC1_4);
legend('no hiding','hiding');
xlabel('Inverse cost of hiding')
title('MPC of C1');
subplot(2,2,3);
plot(deltavar, MPC2_fb_4, deltavar, MPC2_4);
legend('no hiding','hiding');
xlabel('Inverse cost of hiding')
title('MPC of C2');
subplot(2,2,4);
plot(deltavar, MPCQ_fb_4, deltavar, MPCQ_4);
xlabel('Inverse cost of hiding')
legend('no hiding','hiding');
title('MPC of Q');

%% simulation 5: vary agent 2's income
MPP_5 = zeros(1,n);
MPC1_fb_5 = zeros(1,n);
MPC1_5 = zeros(1,n);
MPC2_fb_5 = zeros(1,n);
MPC2_5 = zeros(1,n);
MPCQ_fb_5 = zeros(1,n);
MPCQ_5 = zeros(1,n);
for i = 1:n
    [MPP_5(i), MPC1_fb_5(i), MPC1_5(i), MPC2_fb_5(i), MPC2_5(i), MPCQ_fb_5(i), MPCQ_5(i)] = infinitegame(a1, a2, alpha, beta, delta, y2var(i));
end

% graph
figure;
subplot(2,2,1);
plot(y2var, ones(1,n), y2var, MPP_5);
legend('no hiding','hiding');
xlabel('Income of Agent 2')
title('Marginal Propensity to Report');
subplot(2,2,2);
plot(y2var, MPC1_fb_5, y2var, MPC1_5);
legend('no hiding','hiding');
xlabel('Income of Agent 2')
title('MPC of C1');
subplot(2,2,3);
plot(y2var, MPC2_fb_5, y2var, MPC2_5);
legend('no hiding','hiding');
xlabel('Income of Agent 2')
title('MPC of C2');
subplot(2,2,4);
plot(y2var, MPCQ_fb_5, y2var, MPCQ_5);
xlabel('Income of Agent 2')
legend('no hiding','hiding');
title('MPC of Q');