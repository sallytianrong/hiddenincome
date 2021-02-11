close all; clear all; clc;

%% Parameters
% number of simulations
n = 30;

% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;
a1var = linspace(0.1,0.9,n);

% Two income processes: uniform
y1min = 1; y1max = 5;
y2min = 1; y2max = 5;
y1minvar = linspace(0.1,2,n);
y1maxvar = linspace(4,6,n);
y1maxvar2 = linspace(6,4,n);

% Pareto weight
alpha = 0.5;
alphavar = linspace(0.1,0.9,n);

% Time discounting
beta1 = 0.9;
beta2 = 0.9;
betavar = linspace(0.8,0.97,n);

% Cost of hiding
delta1 = 0.8;
delta2 = 0.8;
deltavar = linspace(0.1,1,n);

%% set up matrices
mean_MPP1 = zeros(1,n);
mean_MPP2 = zeros(1,n);
mean_MPC1_inc1 = zeros(1,n);
mean_MPC1_inc2 = zeros(1,n);
mean_MPC2_inc1 = zeros(1,n);
mean_MPC2_inc2 = zeros(1,n);
mean_MPCQ_inc1 = zeros(1,n);
mean_MPCQ_inc2 = zeros(1,n);
mean_MPC1_inc1_fb = zeros(1,n);
mean_MPC1_inc2_fb = zeros(1,n);
mean_MPC2_inc1_fb = zeros(1,n);
mean_MPC2_inc2_fb = zeros(1,n);
mean_MPCQ_inc1_fb = zeros(1,n);
mean_MPCQ_inc2_fb = zeros(1,n);

%% simulation 1: vary preference of private good
for i = 1:n
    [mean_MPP1(i), mean_MPP2(i), mean_MPC1_inc1(i), mean_MPC1_inc2(i), mean_MPC2_inc1(i), mean_MPC2_inc2(i), mean_MPCQ_inc1(i), mean_MPCQ_inc2(i),...
    mean_MPC1_inc1_fb(i), mean_MPC1_inc2_fb(i), mean_MPC2_inc1_fb(i), mean_MPC2_inc2_fb(i), mean_MPCQ_inc1_fb(i), mean_MPCQ_inc2_fb(i)] =...
    infinitegame_twosides(a1var(i),a2,y1min,y1max,y2min,y2max,alpha,beta1,beta2,delta1,delta2);
end

%% graph
figure;
subplot(2,4,7);
plot(a1var, ones(1,n), a1var, mean_MPP1, a1var, mean_MPP2);
legend('no hiding','agent 1','agent 2');
xlabel('Preference of private good of Agent 1')
title('Marginal Propensity to Report');

subplot(2,4,1);
plot(a1var, mean_MPC1_inc1, a1var, mean_MPC1_inc1_fb);
legend('no hiding','hiding');
xlabel('Preference of private good of Agent 1')
title('MPC of C1 wrt Income of Agent 1');
subplot(2,4,2);
plot(a1var, mean_MPC1_inc2, a1var, mean_MPC1_inc2_fb);
legend('no hiding','hiding');
xlabel('Preference of private good of Agent 1')
title('MPC of C1 wrt Income of Agent 2');

subplot(2,4,3);
plot(a1var, mean_MPC2_inc1, a1var, mean_MPC2_inc1_fb);
legend('no hiding','hiding');
xlabel('Preference of private good of Agent 1')
title('MPC of C2 wrt Income of Agent 1');
subplot(2,4,4);
plot(a1var, mean_MPC2_inc2, a1var, mean_MPC2_inc2_fb);
legend('no hiding','hiding');
xlabel('Preference of private good of Agent 1')
title('MPC of C2 wrt Income of Agent 1');

subplot(2,4,5);
plot(a1var, mean_MPCQ_inc1, a1var, mean_MPCQ_inc1_fb);
legend('no hiding','hiding');
xlabel('Preference of private good of Agent 1')
title('MPC of Q wrt Income of Agent 1');
subplot(2,4,6);
plot(a1var, mean_MPCQ_inc2, a1var, mean_MPCQ_inc2_fb);
legend('no hiding','hiding');
xlabel('Preference of private good of Agent 1')
title('MPC of Q wrt Income of Agent 1');

%% simulation 2: vary mean income of agent 1
for i = 1:n
    [mean_MPP1(i), mean_MPP2(i), mean_MPC1_inc1(i), mean_MPC1_inc2(i), mean_MPC2_inc1(i), mean_MPC2_inc2(i), mean_MPCQ_inc1(i), mean_MPCQ_inc2(i),...
    mean_MPC1_inc1_fb(i), mean_MPC1_inc2_fb(i), mean_MPC2_inc1_fb(i), mean_MPC2_inc2_fb(i), mean_MPCQ_inc1_fb(i), mean_MPCQ_inc2_fb(i)] =...
    infinitegame_twosides(a1,a2,y1minvar(i),y1maxvar(i),y2min,y2max,alpha,beta1,beta2,delta1,delta2);
end

%% graph
figure;
meany1 = (y1minvar + y1maxvar)./2;

subplot(2,4,7);
plot(meany1, ones(1,n), meany1, mean_MPP1, meany1, mean_MPP2);
legend('no hiding','agent 1','agent 2');
xlabel('Mean Income of Agent 1')
title('Marginal Propensity to Report');

subplot(2,4,1);
plot(meany1, mean_MPC1_inc1, meany1, mean_MPC1_inc1_fb);
legend('no hiding','hiding');
xlabel('Mean Income of Agent 1')
title('MPC of C1 wrt Income of Agent 1');
subplot(2,4,2);
plot(meany1, mean_MPC1_inc2, meany1, mean_MPC1_inc2_fb);
legend('no hiding','hiding');
xlabel('Mean Income of Agent 1')
title('MPC of C1 wrt Income of Agent 2');

subplot(2,4,3);
plot(meany1, mean_MPC2_inc1, meany1, mean_MPC2_inc1_fb);
legend('no hiding','hiding');
xlabel('Mean Income of Agent 1')
title('MPC of C2 wrt Income of Agent 1');
subplot(2,4,4);
plot(meany1, mean_MPC2_inc2, meany1, mean_MPC2_inc2_fb);
legend('no hiding','hiding');
xlabel('Mean Income of Agent 1')
title('MPC of C2 wrt Income of Agent 1');

subplot(2,4,5);
plot(meany1, mean_MPCQ_inc1, meany1, mean_MPCQ_inc1_fb);
legend('no hiding','hiding');
xlabel('Mean Income of Agent 1')
title('MPC of Q wrt Income of Agent 1');
subplot(2,4,6);
plot(meany1, mean_MPCQ_inc2, meany1, mean_MPCQ_inc2_fb);
legend('no hiding','hiding');
xlabel('Mean Income of Agent 1')
title('MPC of Q wrt Income of Agent 1');

%% simulation 3: vary variance of income of agent 1
for i = 1:n
    [mean_MPP1(i), mean_MPP2(i), mean_MPC1_inc1(i), mean_MPC1_inc2(i), mean_MPC2_inc1(i), mean_MPC2_inc2(i), mean_MPCQ_inc1(i), mean_MPCQ_inc2(i),...
    mean_MPC1_inc1_fb(i), mean_MPC1_inc2_fb(i), mean_MPC2_inc1_fb(i), mean_MPC2_inc2_fb(i), mean_MPCQ_inc1_fb(i), mean_MPCQ_inc2_fb(i)] =...
    infinitegame_twosides(a1,a2,y1minvar(i),y1maxvar2(i),y2min,y2max,alpha,beta1,beta2,delta1,delta2);
end

%% graph
figure;
vary1 = ((y1maxvar2 - y1minvar).^2)./12;

subplot(2,4,7);
plot(vary1, ones(1,n), vary1, mean_MPP1, vary1, mean_MPP2);
legend('no hiding','agent 1','agent 2');
xlabel('Income Variability of Agent 1')
title('Marginal Propensity to Report');

subplot(2,4,1);
plot(vary1, mean_MPC1_inc1, vary1, mean_MPC1_inc1_fb);
legend('no hiding','hiding');
xlabel('Income Variability of Agent 1')
title('MPC of C1 wrt Income of Agent 1');
subplot(2,4,2);
plot(vary1, mean_MPC1_inc2, vary1, mean_MPC1_inc2_fb);
legend('no hiding','hiding');
xlabel('Income Variability of Agent 1')
title('MPC of C1 wrt Income of Agent 2');

subplot(2,4,3);
plot(vary1, mean_MPC2_inc1, vary1, mean_MPC2_inc1_fb);
legend('no hiding','hiding');
xlabel('Income Variability of Agent 1')
title('MPC of C2 wrt Income of Agent 1');
subplot(2,4,4);
plot(vary1, mean_MPC2_inc2, vary1, mean_MPC2_inc2_fb);
legend('no hiding','hiding');
xlabel('Income Variability of Agent 1')
title('MPC of C2 wrt Income of Agent 1');

subplot(2,4,5);
plot(vary1, mean_MPCQ_inc1, vary1, mean_MPCQ_inc1_fb);
legend('no hiding','hiding');
xlabel('Income Variability of Agent 1')
title('MPC of Q wrt Income of Agent 1');
subplot(2,4,6);
plot(vary1, mean_MPCQ_inc2, vary1, mean_MPCQ_inc2_fb);
legend('no hiding','hiding');
xlabel('Income Variability of Agent 1')
title('MPC of Q wrt Income of Agent 1');

%% simulation 4: vary Pareto weights
for i = 1:n
    [mean_MPP1(i), mean_MPP2(i), mean_MPC1_inc1(i), mean_MPC1_inc2(i), mean_MPC2_inc1(i), mean_MPC2_inc2(i), mean_MPCQ_inc1(i), mean_MPCQ_inc2(i),...
    mean_MPC1_inc1_fb(i), mean_MPC1_inc2_fb(i), mean_MPC2_inc1_fb(i), mean_MPC2_inc2_fb(i), mean_MPCQ_inc1_fb(i), mean_MPCQ_inc2_fb(i)] =...
    infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alphavar(i),beta1,beta2,delta1,delta2);
end

%% graph
figure;

subplot(2,4,7);
plot(alphavar, ones(1,n), alphavar, mean_MPP1, alphavar, mean_MPP2);
legend('no hiding','agent 1','agent 2');
xlabel('Pareto Weight of Agent 1')
title('Marginal Propensity to Report');

subplot(2,4,1);
plot(alphavar, mean_MPC1_inc1, alphavar, mean_MPC1_inc1_fb);
legend('no hiding','hiding');
xlabel('Pareto Weight of Agent 1')
title('MPC of C1 wrt Income of Agent 1');
subplot(2,4,2);
plot(alphavar, mean_MPC1_inc2, alphavar, mean_MPC1_inc2_fb);
legend('no hiding','hiding');
xlabel('Pareto Weight of Agent 1')
title('MPC of C1 wrt Income of Agent 2');

subplot(2,4,3);
plot(alphavar, mean_MPC2_inc1, alphavar, mean_MPC2_inc1_fb);
legend('no hiding','hiding');
xlabel('Pareto Weight of Agent 1')
title('MPC of C2 wrt Income of Agent 1');
subplot(2,4,4);
plot(alphavar, mean_MPC2_inc2, alphavar, mean_MPC2_inc2_fb);
legend('no hiding','hiding');
xlabel('Pareto Weight of Agent 1')
title('MPC of C2 wrt Income of Agent 1');

subplot(2,4,5);
plot(alphavar, mean_MPCQ_inc1, alphavar, mean_MPCQ_inc1_fb);
legend('no hiding','hiding');
xlabel('Pareto Weight of Agent 1')
title('MPC of Q wrt Income of Agent 1');
subplot(2,4,6);
plot(alphavar, mean_MPCQ_inc2, alphavar, mean_MPCQ_inc2_fb);
legend('no hiding','hiding');
xlabel('Pareto Weight of Agent 1')
title('MPC of Q wrt Income of Agent 1');

%% simulation 5: vary time discounting factor
for i = 1:n
    [mean_MPP1(i), mean_MPP2(i), mean_MPC1_inc1(i), mean_MPC1_inc2(i), mean_MPC2_inc1(i), mean_MPC2_inc2(i), mean_MPCQ_inc1(i), mean_MPCQ_inc2(i),...
    mean_MPC1_inc1_fb(i), mean_MPC1_inc2_fb(i), mean_MPC2_inc1_fb(i), mean_MPC2_inc2_fb(i), mean_MPCQ_inc1_fb(i), mean_MPCQ_inc2_fb(i)] =...
    infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,betavar(i),beta2,delta1,delta2);
    i
end

%% graph
figure;

subplot(2,4,7);
plot(betavar, ones(1,n), betavar, mean_MPP1, betavar, mean_MPP2);
legend('no hiding','agent 1','agent 2');
xlabel('Time Discounting Factor')
title('Marginal Propensity to Report');

subplot(2,4,1);
plot(betavar, mean_MPC1_inc1, betavar, mean_MPC1_inc1_fb);
legend('no hiding','hiding');
xlabel('Time Discounting Factor')
title('MPC of C1 wrt Income of Agent 1');
subplot(2,4,2);
plot(betavar, mean_MPC1_inc2, betavar, mean_MPC1_inc2_fb);
legend('no hiding','hiding');
xlabel('Time Discounting Factor')
title('MPC of C1 wrt Income of Agent 2');

subplot(2,4,3);
plot(betavar, mean_MPC2_inc1, betavar, mean_MPC2_inc1_fb);
legend('no hiding','hiding');
xlabel('Time Discounting Factor')
title('MPC of C2 wrt Income of Agent 1');
subplot(2,4,4);
plot(betavar, mean_MPC2_inc2, betavar, mean_MPC2_inc2_fb);
legend('no hiding','hiding');
xlabel('Time Discounting Factor')
title('MPC of C2 wrt Income of Agent 1');

subplot(2,4,5);
plot(betavar, mean_MPCQ_inc1, betavar, mean_MPCQ_inc1_fb);
legend('no hiding','hiding');
xlabel('Time Discounting Factor')
title('MPC of Q wrt Income of Agent 1');
subplot(2,4,6);
plot(betavar, mean_MPCQ_inc2, betavar, mean_MPCQ_inc2_fb);
legend('no hiding','hiding');
xlabel('Time Discounting Factor')
title('MPC of Q wrt Income of Agent 1');

%% simulation 6: vary cost of hiding
for i = 1:n
    [mean_MPP1(i), mean_MPP2(i), mean_MPC1_inc1(i), mean_MPC1_inc2(i), mean_MPC2_inc1(i), mean_MPC2_inc2(i), mean_MPCQ_inc1(i), mean_MPCQ_inc2(i),...
    mean_MPC1_inc1_fb(i), mean_MPC1_inc2_fb(i), mean_MPC2_inc1_fb(i), mean_MPC2_inc2_fb(i), mean_MPCQ_inc1_fb(i), mean_MPCQ_inc2_fb(i)] =...
    infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,beta1,beta2,deltavar(i),delta2);
end

%% graph
figure;

subplot(2,4,7);
plot(deltavar, ones(1,n), deltavar, mean_MPP1, deltavar, mean_MPP2);
legend('no hiding','agent 1','agent 2');
xlabel('(Inverse) Cost of Hiding of Agent 1')
title('Marginal Propensity to Report');

subplot(2,4,1);
plot(deltavar, mean_MPC1_inc1, deltavar, mean_MPC1_inc1_fb);
legend('no hiding','hiding');
xlabel('(Inverse) Cost of Hiding of Agent 1')
title('MPC of C1 wrt Income of Agent 1');
subplot(2,4,2);
plot(deltavar, mean_MPC1_inc2, deltavar, mean_MPC1_inc2_fb);
legend('no hiding','hiding');
xlabel('(Inverse) Cost of Hiding of Agent 1')
title('MPC of C1 wrt Income of Agent 2');

subplot(2,4,3);
plot(deltavar, mean_MPC2_inc1, deltavar, mean_MPC2_inc1_fb);
legend('no hiding','hiding');
xlabel('(Inverse) Cost of Hiding of Agent 1')
title('MPC of C2 wrt Income of Agent 1');
subplot(2,4,4);
plot(deltavar, mean_MPC2_inc2, deltavar, mean_MPC2_inc2_fb);
legend('no hiding','hiding');
xlabel('(Inverse) Cost of Hiding of Agent 1')
title('MPC of C2 wrt Income of Agent 1');

subplot(2,4,5);
plot(deltavar, mean_MPCQ_inc1, deltavar, mean_MPCQ_inc1_fb);
legend('no hiding','hiding');
xlabel('(Inverse) Cost of Hiding of Agent 1')
title('MPC of Q wrt Income of Agent 1');
subplot(2,4,6);
plot(deltavar, mean_MPCQ_inc2, deltavar, mean_MPCQ_inc2_fb);
legend('no hiding','hiding');
xlabel('(Inverse) Cost of Hiding of Agent 1')
title('MPC of Q wrt Income of Agent 1');
