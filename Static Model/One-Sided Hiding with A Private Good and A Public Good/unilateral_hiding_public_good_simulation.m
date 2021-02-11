% Unilateral Hiding Simulation with Public Good
% Sally Zhang
% July 8, 2020
% Simulation of a two agent static model where only one agent can hide. One
% private good and one public good.

clear all; close all; clc;
rng(0);

%% Set Up
% Parameters
n = 30;
% cobb-douglas utility function parameter
% utility = a*log(private good) + (1-a)*log(public good)
a1 = 0.5;
a2 = 0.5;
avar = linspace(0.1,0.9,n);
% price for the public good (price of the private good is
% normalized to 1)
price = 1;
pricevar = linspace(0.75,1.25,n);
% delta [0,1] is cost of hiding. delta=1 means there is no cost of hiding.
% delta=0 means cost of hiding is prohibitive.
delta0=0.9;
delta = linspace(0.1,0.9,n);
% alpha [0,1] is the Pareto weight of agent 1.
alpha0=0.3;
alpha = linspace(0.1,0.9,n);
% Agent 2's income is normalized to 1. Agent 1's
% income are y0, y0+rho, y0+2rho.
y0 = 1;
yvar = linspace(0.5,2,n);
rho0 = 1;
rho = linspace(0.1,1.9,n);
% Generate an income process with increasing variance but constant mean
ycons = 2 - rho;
% p is the probability of states
p0 = [0.33 0.33 0.33];
% probability of states
p = zeros(n,3);
p(:,2) = linspace(0.1,0.9,n)';
p(:,1) = (1-p(:,2))./2;
p(:,3) = p(:,1);


%% Simulations
% First simulation: vary cost of hiding
c1 = zeros(3,3,n);
d1 = zeros(3,3,n);
Q1 = zeros(3,3,n);
eu1 = zeros(n,3);
ev1 = zeros(n,3);
eh1 = zeros(n,3);
esp1 = zeros(n,3);
ems1 = zeros(n,3);
us1 = zeros(n,3);
vs1 = zeros(n,3);
for i = 1:n
    tic
    [c1(:,:,i), d1(:,:,i), Q1(:,:,i), eu1(i,:), ev1(i,:), eh1(i,:), esp1(i,:), ems1(i,:), us1(i,:), vs1(i,:)] = unilateral_hiding_public_good(a1,a2,price,delta(i),alpha0,y0,rho0,p0);
    toc
end

%% Second simulation: vary Pareto weights
c2 = zeros(3,3,n);
d2 = zeros(3,3,n);
Q2 = zeros(3,3,n);
eu2 = zeros(n,3);
ev2 = zeros(n,3);
eh2 = zeros(n,3);
esp2 = zeros(n,3);
ems2 = zeros(n,3);
us2 = zeros(n,3);
vs2 = zeros(n,3);
for i = 1:n
    [c2(:,:,i), d2(:,:,i), Q2(:,:,i), eu2(i,:), ev2(i,:), eh2(i,:), esp2(i,:), ems2(i,:), us2(i,:), vs2(i,:)] = unilateral_hiding_public_good(a1,a2,price,delta0,alpha(i),y0,rho0,p0);
end

%% Third simulation: vary probability of state 2
c3 = zeros(3,3,n);
d3 = zeros(3,3,n);
Q3 = zeros(3,3,n);
eu3 = zeros(n,3);
ev3 = zeros(n,3);
eh3 = zeros(n,3);
esp3 = zeros(n,3);
ems3 = zeros(n,3);
us3 = zeros(n,3);
vs3 = zeros(n,3);
for i = 1:n
    [c3(:,:,i), d3(:,:,i), Q3(:,:,i), eu3(i,:), ev3(i,:), eh3(i,:), esp3(i,:), ems3(i,:), us3(i,:), vs3(i,:)] = unilateral_hiding_public_good(a1,a2,price,delta0,alpha0,y0,rho0,p(i,:));
end

%% Fourth simulation: vary income
c4 = zeros(3,3,n);
d4 = zeros(3,3,n);
Q4 = zeros(3,3,n);
eu4 = zeros(n,3);
ev4 = zeros(n,3);
eh4 = zeros(n,3);
esp4 = zeros(n,3);
ems4 = zeros(n,3);
us4 = zeros(n,3);
vs4 = zeros(n,3);
for i = 1:n
    [c4(:,:,i), d4(:,:,i), Q4(:,:,i), eu4(i,:), ev4(i,:), eh4(i,:), esp4(i,:), ems4(i,:), us4(i,:), vs4(i,:)] = unilateral_hiding_public_good(a1,a2,price,delta0,alpha0,yvar(i),rho0,p0);
end

%% Fifth simulation: vary price of public good
c5 = zeros(3,3,n);
d5 = zeros(3,3,n);
Q5 = zeros(3,3,n);
eu5 = zeros(n,3);
ev5 = zeros(n,3);
eh5 = zeros(n,3);
esp5 = zeros(n,3);
ems5 = zeros(n,3);
us5 = zeros(n,3);
vs5 = zeros(n,3);
for i = 1:n
    [c5(:,:,i), d5(:,:,i), Q5(:,:,i), eu5(i,:), ev5(i,:), eh5(i,:), esp5(i,:), ems5(i,:), us5(i,:), vs5(i,:)] = unilateral_hiding_public_good(a1,a2,pricevar(i),delta0,alpha0,y0,rho0,p0);
end

%% Sixth simulation: vary preference of public good
c6 = zeros(3,3,n);
d6 = zeros(3,3,n);
Q6 = zeros(3,3,n);
eu6 = zeros(n,3);
ev6 = zeros(n,3);
eh6 = zeros(n,3);
esp6 = zeros(n,3);
ems6 = zeros(n,3);
us6 = zeros(n,3);
vs6 = zeros(n,3);
for i = 1:n
    [c6(:,:,i), d6(:,:,i), Q6(:,:,i), eu6(i,:), ev6(i,:), eh6(i,:), esp6(i,:), ems6(i,:), us6(i,:), vs6(i,:)] = unilateral_hiding_public_good(avar(i),avar(i),price,delta0,alpha0,y0,rho0,p0);
end

%% Seventh simulation: vary income variability
c7 = zeros(3,3,n);
d7 = zeros(3,3,n);
Q7 = zeros(3,3,n);
eu7 = zeros(n,3);
ev7 = zeros(n,3);
eh7 = zeros(n,3);
esp7 = zeros(n,3);
ems7 = zeros(n,3);
us7 = zeros(n,3);
vs7 = zeros(n,3);
for i = 1:n
    [c7(:,:,i), d7(:,:,i), Q7(:,:,i), eu7(i,:), ev7(i,:), eh7(i,:), esp7(i,:), ems7(i,:), us7(i,:), vs7(i,:)] = unilateral_hiding_public_good(a1,a2,price,delta0,alpha0,ycons(i),rho(i),p0);
end

%% Plot results
% line of zero
zero = zeros(1,n);

% First figure varies the cost of hiding
figure(1);
subplot(2,4,1);
plot(delta,eu1(:,1),delta,eu1(:,2),delta,eu1(:,3))
title('Expected Utility of Agent 1');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(delta,ev1(:,1),delta,ev1(:,2),delta,ev1(:,3))
title('Expected Utility of Agent 2');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(delta,eh1(:,1),delta,eh1(:,2),delta,eh1(:,3))
title('Expected Utility of Household');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(delta,esp1(:,1),delta,esp1(:,2),delta,esp1(:,3))
title('Equally Weighted Utility of Household');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(delta,us1(:,1),delta,us1(:,2),delta,us1(:,3),delta,zero);
title('Share of Surplus of Agent 1');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Share');
ymax = max(max(us1));
ymin = min(min(us1));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,6);
plot(delta,vs1(:,1),delta,vs1(:,2),delta,vs1(:,3),delta,zero);
title('Share of Surplus of Agent 2');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Share');
ymax = max(max(vs1));
ymin = min(min(vs1));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,7);
plot(delta,ems1(:,1),delta,ems1(:,2),delta,ems1(:,3));
title('Total Marriage Surplus');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.3 0.1 0.1]);

% Second figure varies agent 1's Pareto weight
figure(2);
subplot(2,4,1);
plot(alpha,eu2(:,1),alpha,eu2(:,2),alpha,eu2(:,3))
title('Expected Utility of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(alpha,ev2(:,1),alpha,ev2(:,2),alpha,ev2(:,3))
title('Expected Utility of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(alpha,eh2(:,1),alpha,eh2(:,2),alpha,eh2(:,3))
title('Expected Utility of Household');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(alpha,esp2(:,1),alpha,esp2(:,2),alpha,esp2(:,3))
title('Equally Weighted Utility of Household');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(alpha,us2(:,1),alpha,us2(:,2),alpha,us2(:,3),alpha,zero);
title('Share of Surplus of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Share');
ymax = max(max(us2));
ymin = min(min(us2));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,6);
plot(alpha,vs2(:,1),alpha,vs2(:,2),alpha,vs2(:,3),alpha,zero);
title('Share of Surplus of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Share');
ymax = max(max(vs2));
ymin = min(min(vs2));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,7);
plot(alpha,ems2(:,1),alpha,ems2(:,2),alpha,ems2(:,3))
title('Marriage Surplus');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.3 0.1 0.1]);

% Third figure varies the likelihood of state 2
figure(3);
subplot(2,4,1);
plot(p(:,2),eu3(:,1),p(:,2),eu3(:,2),p(:,2),eu3(:,3));
title('Expected Utility of Agent 1');
xlabel('Probability of State 2');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(p(:,2),ev3(:,1),p(:,2),ev3(:,2),p(:,2),ev3(:,3));
title('Expected Utility of Agent 2');
xlabel('Probability of State 2');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(p(:,2),eh3(:,1),p(:,2),eh3(:,2),p(:,2),eh3(:,3));
title('Expected Utility of Household');
xlabel('Probability of State 2');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(p(:,2),esp3(:,1),p(:,2),esp3(:,2),p(:,2),esp3(:,3));
title('Equally Weighted Utility of Household');
xlabel('Probability of State 2');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(p(:,2),us3(:,1),p(:,2),us3(:,2),p(:,2),us3(:,3),p(:,2),zero);
title('Share of Surplus of Agent 1');
xlabel('Probability of State 2');
ylabel('Share');
ymax = max(max(us3));
ymin = min(min(us3));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,6);
plot(p(:,2),vs3(:,1),p(:,2),vs3(:,2),p(:,2),vs3(:,3),p(:,2),zero);
title('Share of Surplus of Agent 2');
xlabel('Probability of State 2');
ylabel('Share');
ymax = max(max(vs3));
ymin = min(min(vs3));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,7);
plot(p(:,2),ems3(:,1),p(:,2),ems3(:,2),p(:,2),ems3(:,3))
title('Marriage Surplus');
xlabel('Probability of State 2');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.3 0.1 0.1]);

% Fourth figure varies income
yvar = yvar+2;
figure(4);
subplot(2,4,1);
plot(yvar,eu4(:,1),yvar,eu4(:,2),yvar,eu4(:,3));
title('Expected Utility of Agent 1');
xlabel('Expected Household Income');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(yvar,ev4(:,1),yvar,ev4(:,2),yvar,ev4(:,3));
title('Expected Utility of Agent 2');
xlabel('Expected Household Income');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(yvar,eh4(:,1),yvar,eh4(:,2),yvar,eh4(:,3));
title('Expected Utility of Household');
xlabel('Expected Household Income');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(yvar,esp4(:,1),yvar,esp4(:,2),yvar,esp4(:,3));
title('Equally Weighted Utility of Household');
xlabel('Expected Household Income');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(yvar,us4(:,1),yvar,us4(:,2),yvar,us4(:,3),yvar,zero);
title('Share of Surplus of Agent 1');
xlabel('Expected Household Income');
ylabel('Share');
ymax = max(max(us4));
ymin = min(min(us4));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,6);
plot(yvar,vs4(:,1),yvar,vs4(:,2),yvar,vs4(:,3),yvar,zero);
title('Share of Surplus of Agent 2');
xlabel('Expected Household Income');
ylabel('Share');
ymax = max(max(vs4));
ymin = min(min(vs4));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,7);
plot(yvar,ems4(:,1),yvar,ems4(:,2),yvar,ems4(:,3))
title('Total Marriage Surplus');
xlabel('Expected Household Income');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.3 0.1 0.1]);

% Fifth figure varies price
figure(5);
subplot(2,4,1);
plot(pricevar,eu5(:,1),pricevar,eu5(:,2),pricevar,eu5(:,3));
title('Expected Utility of Agent 1');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(pricevar,ev5(:,1),pricevar,ev5(:,2),pricevar,ev5(:,3));
title('Expected Utility of Agent 2');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(pricevar,eh5(:,1),pricevar,eh5(:,2),pricevar,eh5(:,3));
title('Expected Utility of Household');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(pricevar,esp5(:,1),pricevar,esp5(:,2),pricevar,esp5(:,3));
title('Equally Weighted Utility of Household');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(pricevar,us5(:,1),pricevar,us5(:,2),pricevar,us5(:,3),pricevar,zero);
title('Share of Surplus of Agent 1');
xlabel('Price of Public Good');
ylabel('Share');
ymax = max(max(us5));
ymin = min(min(us5));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,6);
plot(pricevar,vs5(:,1),pricevar,vs5(:,2),pricevar,vs5(:,3),pricevar,zero);
title('Share of Surplus of Agent 2');
xlabel('Price of Public Good');
ylabel('Share');
ymax = max(max(vs5));
ymin = min(min(vs5));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,7);
plot(pricevar,ems5(:,1),pricevar,ems5(:,2),pricevar,ems5(:,3))
title('Marriage Surplus');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.3 0.1 0.1]);

% Sixth figure varies preference
figure(6);
subplot(2,4,1);
plot(avar,eu6(:,1),avar,eu6(:,2),avar,eu6(:,3));
title('Expected Utility of Agent 1');
xlabel('Preference of Private Good');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(avar,ev6(:,1),avar,ev6(:,2),avar,ev6(:,3));
title('Expected Utility of Agent 2');
xlabel('Preference of Private Good');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(avar,eh6(:,1),avar,eh6(:,2),avar,eh6(:,3));
title('Expected Utility of Household');
xlabel('Preference of Private Good');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(avar,esp6(:,1),avar,esp6(:,2),avar,esp6(:,3));
title('Equally Weighted Utility of Household');
xlabel('Preference of Private Good');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(avar,us6(:,1),avar,us6(:,2),avar,us6(:,3),avar,zero);
title('Share of Surplus of Agent 1');
xlabel('Preference of Private Good');
ylabel('Share');
ymax = max(max(us6));
ymin = min(min(us6));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,6);
plot(avar,vs6(:,1),avar,vs6(:,2),avar,vs6(:,3),avar,zero);
title('Share of Surplus of Agent 2');
xlabel('Preference of Private Good');
ylabel('Share');
ymax = max(max(vs6));
ymin = min(min(vs6));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,7);
plot(avar,ems6(:,1),avar,ems6(:,2),avar,ems6(:,3))
title('Total Marriage Surplus');
xlabel('Preference of Private Good');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.3 0.1 0.1]);

% Seventh figure varies income variability
figure(7);
subplot(2,4,1);
plot(rho,eu7(:,1),rho,eu7(:,2),rho,eu7(:,3));
title('Expected Utility of Agent 1');
xlabel('Income Variability');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(rho,ev7(:,1),rho,ev7(:,2),rho,ev7(:,3));
title('Expected Utility of Agent 2');
xlabel('Income Variability');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(rho,eh7(:,1),rho,eh7(:,2),rho,eh7(:,3));
title('Expected Utility of Household');
xlabel('Income Variability');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(rho,esp7(:,1),rho,esp7(:,2),rho,esp7(:,3));
title('Equally Weighted Utility of Household');
xlabel('Income Variability');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(rho,us7(:,1),rho,us7(:,2),rho,us7(:,3),rho,zero);
title('Share of Surplus of Agent 1');
xlabel('Income Variability');
ylabel('Share');
ymax = max(max(us7));
ymin = min(min(us7));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,6);
plot(rho,vs7(:,1),rho,vs7(:,2),rho,vs7(:,3),rho,zero);
title('Share of Surplus of Agent 2');
xlabel('Income Variability');
ylabel('Share');
ymax = max(max(vs7));
ymin = min(min(vs7));
ylim([ymin ymax]);
legend('hide');

subplot(2,4,7);
plot(rho,ems7(:,1),rho,ems7(:,2),rho,ems7(:,3))
title('Total Marriage Surplus');
xlabel('Income Variability');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.3 0.1 0.1]);
