% Bilateral Hiding with Observable and Unobservable Consumption
% Introducing observable and unobservable goods. For right now there is
% only one observable good and one unobservable.
% May 13, 2020

clear all; close all; clc;

% Parameters
% cobb-douglas utility function parameter
% utility = a*log(observable) + (1-a)*log(unobservable)
a1 = 0.5;
a2 = 0.5;
a = linspace(0.1,0.9,10);
% price for the unobservable good (price of the observable good is
% normalized to 1)
price0 = 1.5;
price = linspace(0.5,2,10);
% 4 hiding parameters: for two goods and two agents
% 1 unit of hidden income can be used as delta expenditure toward a good
% observable delta < unobservable delta
delta1o = 0.8;
delta2o = 0.8;
delta1u = 0.9;
delta2u = 0.9;
deltau = linspace(0.1,1,5);
% alpha [0,1] is the Pareto weight of agent 1.
alpha0 = 0.3;
alpha = linspace(0.1,0.9,10);
% Agent 2's income is normalized to 0,1,2 in the three states. Agent 1's
% income are y0, y0+rho, y0+2rho.
y0 = 1;
rho0 = 1;
rho = linspace(0.1,1.9,10);
% Generate an income process with increasing variance but constant mean
ycons = 2 - rho;
% p is the probability of states for agent 1, q is the probability of
% states for agent 2;
p = [0.33 0.33 0.33];
q = [0.33;0.33;0.33];

%% Simulations
% First simulation: vary cost of hiding
n = 5;
eco1 = zeros(n,5);
ecu1 = zeros(n,5);
edo1 = zeros(n,5);
edu1 = zeros(n,5);
etoto1 = zeros(n,5);
etotu1 = zeros(n,5);
eu1 = zeros(n,5);
ev1 = zeros(n,5);
eh1 = zeros(n,5);
esp1 = zeros(n,5);
for i = 1:n
    tic
    [eco1(i,:), ecu1(i,:), edo1(i,:), edu1(i,:), etoto1(i,:), etotu1(i,:), eu1(i,:), ev1(i,:), eh1(i,:), esp1(i,:)] = bilateral_hiding_twogoods(a1, a2, price0, delta1o, deltau(i), delta2o, deltau(i), alpha0, y0, rho0, p, q);
    toc
end

%% Second simulation: vary Pareto weights
eco2 = zeros(10,5);
ecu2 = zeros(10,5);
edo2 = zeros(10,5);
edu2 = zeros(10,5);
etoto2 = zeros(10,5);
etotu2 = zeros(10,5);
eu2 = zeros(10,5);
ev2 = zeros(10,5);
eh2 = zeros(10,5);
esp2 = zeros(10,5);
for i = 1:10
    tic
    [eco2(i,:), ecu2(i,:), edo2(i,:), edu2(i,:), etoto2(i,:), etotu2(i,:), eu2(i,:), ev2(i,:), eh2(i,:), esp2(i,:)] = bilateral_hiding_twogoods(a1, a2, price0, delta1o, delta2u, delta2o, delta2u, alpha(i), y0, rho0, p, q);
    toc
end

%% Third simulation: vary Cobb-Douglas parameter
eco3 = zeros(10,5);
ecu3 = zeros(10,5);
edo3 = zeros(10,5);
edu3 = zeros(10,5);
etoto3 = zeros(10,5);
etotu3 = zeros(10,5);
eu3 = zeros(10,5);
ev3 = zeros(10,5);
eh3 = zeros(10,5);
esp3 = zeros(10,5);
for i = 1:10
    tic
    [eco3(i,:), ecu3(i,:), edo3(i,:), edu3(i,:), etoto3(i,:), etotu3(i,:), eu3(i,:), ev3(i,:), eh3(i,:), esp3(i,:)] = bilateral_hiding_twogoods(a(i), a(i), price0, delta1o, delta2u, delta2o, delta2u, alpha0, y0, rho0, p, q);
    toc
end

%% Fourth simulation: vary income gaps
eco4 = zeros(10,5);
ecu4 = zeros(10,5);
edo4 = zeros(10,5);
edu4 = zeros(10,5);
etoto4 = zeros(10,5);
etotu4 = zeros(10,5);
eu4 = zeros(10,5);
ev4 = zeros(10,5);
eh4 = zeros(10,5);
esp4 = zeros(10,5);
for i = 1:10
    tic
    [eco4(i,:), ecu4(i,:), edo4(i,:), edu4(i,:), etoto4(i,:), etotu4(i,:), eu4(i,:), ev4(i,:), eh4(i,:), esp4(i,:)] = bilateral_hiding_twogoods(a1, a2, price0, delta1o, delta2u, delta2o, delta2u, alpha0, ycons(i), rho(i), p, q);
    toc
end

%% Fifth simulation: vary price of unobservable good
eco5 = zeros(10,5);
ecu5 = zeros(10,5);
edo5 = zeros(10,5);
edu5 = zeros(10,5);
etoto5 = zeros(10,5);
etotu5 = zeros(10,5);
eu5 = zeros(10,5);
ev5 = zeros(10,5);
eh5 = zeros(10,5);
esp5 = zeros(10,5);
for i = 1:10
    tic
    [eco5(i,:), ecu5(i,:), edo5(i,:), edu5(i,:), etoto5(i,:), etotu5(i,:), eu5(i,:), ev5(i,:), eh5(i,:), esp5(i,:)] = bilateral_hiding_twogoods(a1, a2, price(i), delta1o, delta2u, delta2o, delta2u, alpha0, y0, rho0, p, q);
    toc
end

%% Plot simulation results
% First figure varies the cost of hiding
figure(1);
subplot(2,3,1);
plot(deltau,etoto1(:,1),deltau,etoto1(:,2),deltau,etoto1(:,3),deltau,etoto1(:,4),deltau,etoto1(:,5));
title('Consumption of Observable Good');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Consumption');

subplot(2,3,2);
plot(deltau,etotu1(:,1),deltau,etotu1(:,2),deltau,etotu1(:,3),deltau,etotu1(:,4),deltau,etotu1(:,5));
title('Consumption of Unobservable Good');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Consumption');

subplot(2,3,3);
plot(deltau,eu1(:,1),deltau,eu1(:,2),deltau,eu1(:,3),deltau,eu1(:,4),deltau,eu1(:,5));
title('Expected Utility of Agent 1');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Location','northwest');

subplot(2,3,4);
plot(deltau,ev1(:,1),deltau,ev1(:,2),deltau,ev1(:,3),deltau,ev1(:,4),deltau,ev1(:,5));
title('Expected Utility of Agent 2');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,3,5);
plot(deltau,eh1(:,1),deltau,eh1(:,2),deltau,eh1(:,3),deltau,eh1(:,4),deltau,eh1(:,5));
title('Expected Utility of Household');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,3,6);
plot(deltau,esp1(:,1),deltau,esp1(:,2),deltau,esp1(:,3),deltau,esp1(:,4),deltau,esp1(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

%% Second figure varies Pareto weights
figure(2);
subplot(2,3,1);
plot(alpha,etoto2(:,1),alpha,etoto2(:,2),alpha,etoto2(:,3),alpha,etoto2(:,4),alpha,etoto2(:,5));
title('Consumption of Observable Good');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Consumption');

subplot(2,3,2);
plot(alpha,etotu2(:,1),alpha,etotu2(:,2),alpha,etotu2(:,3),alpha,etotu2(:,4),alpha,etotu2(:,5));
title('Consumption of Unobservable Good');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Consumption');

subplot(2,3,3);
plot(alpha,eu2(:,1),alpha,eu2(:,2),alpha,eu2(:,3),alpha,eu2(:,4),alpha,eu2(:,5));
title('Expected Utility of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Location','northwest');

subplot(2,3,4);
plot(alpha,ev2(:,1),alpha,ev2(:,2),alpha,ev2(:,3),alpha,ev2(:,4),alpha,ev2(:,5));
title('Expected Utility of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,3,5);
plot(alpha,eh2(:,1),alpha,eh2(:,2),alpha,eh2(:,3),alpha,eh2(:,4),alpha,eh2(:,5));
title('Expected Utility of Household');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,3,6);
plot(alpha,esp2(:,1),alpha,esp2(:,2),alpha,esp2(:,3),alpha,esp2(:,4),alpha,esp2(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

%% Third figure varies Cobb-Douglas parameter
figure(3);
subplot(2,3,1);
plot(a,etoto3(:,1),a,etoto3(:,2),a,etoto3(:,3),a,etoto3(:,4),a,etoto3(:,5));
title('Consumption of Observable Good');
xlabel('Cobb Douglas coefficient on observable good');
ylabel('Consumption');

subplot(2,3,2);
plot(a,etotu3(:,1),a,etotu3(:,2),a,etotu3(:,3),a,etotu3(:,4),a,etotu3(:,5));
title('Consumption of Unobservable Good');
xlabel('Cobb Douglas coefficient on observable good');
ylabel('Consumption');

subplot(2,3,4);
plot(a,eu3(:,1),a,eu3(:,2),a,eu3(:,3),a,eu3(:,4),a,eu3(:,5));
title('Expected Utility of Agent 1');
xlabel('Cobb Douglas coefficient on observable good');
ylabel('Utility');
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Location','northwest');

subplot(2,3,5);
plot(a,ev3(:,1),a,ev3(:,2),a,ev3(:,3),a,ev3(:,4),a,ev3(:,5));
title('Expected Utility of Agent 2');
xlabel('Cobb Douglas coefficient on observable good');
ylabel('Utility');
legend('hide');

subplot(2,3,3);
plot(a,eh3(:,1),a,eh3(:,2),a,eh3(:,3),a,eh3(:,4),a,eh3(:,5));
title('Expected Utility of Household');
xlabel('Cobb Douglas coefficient on observable good');
ylabel('Utility');
legend('hide');

subplot(2,3,6);
plot(a,esp3(:,1),a,esp3(:,2),a,esp3(:,3),a,esp3(:,4),a,esp3(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Cobb Douglas coefficient on observable good');
ylabel('Utility');
legend('hide');

%% Fourth figure varies income variability
figure(4);
subplot(2,3,1);
plot(rho,etoto4(:,1),rho,etoto4(:,2),rho,etoto4(:,3),rho,etoto4(:,4),rho,etoto4(:,5));
title('Consumption of Observable Good');
xlabel('Income variability');
ylabel('Consumption');

subplot(2,3,2);
plot(rho,etotu4(:,1),rho,etotu4(:,2),rho,etotu4(:,3),rho,etotu4(:,4),rho,etotu4(:,5));
title('Consumption of Unobservable Good');
xlabel('Income variability');
ylabel('Consumption');

subplot(2,3,4);
plot(rho,eu4(:,1),rho,eu4(:,2),rho,eu4(:,3),rho,eu4(:,4),rho,eu4(:,5));
title('Expected Utility of Agent 1');
xlabel('Income variability');
ylabel('Utility');
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Location','northwest');

subplot(2,3,5);
plot(rho,ev4(:,1),rho,ev4(:,2),rho,ev4(:,3),rho,ev4(:,4),rho,ev4(:,5));
title('Expected Utility of Agent 2');
xlabel('Income variability');
ylabel('Utility');
legend('hide');

subplot(2,3,3);
plot(rho,eh4(:,1),rho,eh4(:,2),rho,eh4(:,3),rho,eh4(:,4),rho,eh4(:,5));
title('Expected Utility of Household');
xlabel('Income variability');
ylabel('Utility');
legend('hide');

subplot(2,3,6);
plot(rho,esp4(:,1),rho,esp4(:,2),rho,esp4(:,3),rho,esp4(:,4),rho,esp4(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Income variability');
ylabel('Utility');
legend('hide');

%% Fifth figure varies price
figure(5);
subplot(2,3,1);
plot(price,etoto5(:,1),price,etoto5(:,2),price,etoto5(:,3),price,etoto5(:,4),price,etoto5(:,5));
title('Consumption of Observable Good');
xlabel('Price of unobservable good');
ylabel('Consumption');

subplot(2,3,2);
plot(price,etotu5(:,1),price,etotu5(:,2),price,etotu5(:,3),price,etotu5(:,4),price,etotu5(:,5));
title('Consumption of Unobservable Good');
xlabel('Price of unobservable good');
ylabel('Consumption');

subplot(2,3,4);
plot(price,eu5(:,1),price,eu5(:,2),price,eu5(:,3),price,eu5(:,4),price,eu5(:,5));
title('Expected Utility of Agent 1');
xlabel('Price of unobservable good');
ylabel('Utility');
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Location','northwest');

subplot(2,3,5);
plot(price,ev5(:,1),price,ev5(:,2),price,ev5(:,3),price,ev5(:,4),price,ev5(:,5));
title('Expected Utility of Agent 2');
xlabel('Price of unobservable good');
ylabel('Utility');
legend('hide');

subplot(2,3,3);
plot(price,eh5(:,1),price,eh5(:,2),price,eh5(:,3),price,eh5(:,4),price,eh5(:,5));
title('Expected Utility of Household');
xlabel('Price of unobservable good');
ylabel('Utility');
legend('hide');

subplot(2,3,6);
plot(price,esp5(:,1),price,esp5(:,2),price,esp5(:,3),price,esp5(:,4),price,esp5(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Price of unobservable good');
ylabel('Utility');
legend('hide');

