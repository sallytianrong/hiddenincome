% Bilateral Hiding Simulation with Public Good
% Sally Zhang
% July 21, 2020
% Simulation of a two agent static model where both agents can hide. One
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
delta1 = 0.9;
delta2 = 0.9;
deltavar = linspace(0.1,0.95,n);
% alpha [0,1] is the Pareto weight of agent 1.
alpha = 0.3;
alphavar = linspace(0.1,0.9,n);
% Agent 2's income is normalized to [1,2,3]. Agent 1's
% income are y0, y0+rho, y0+2rho.
y0 = 1;
yvar = linspace(0.5,2,n);
rho = 1;
rhovar = linspace(0.1,1.9,n);
% Generate an income process with increasing variance but constant mean
ycons = 2 - rhovar;
% p is the probability of states
p = [0.33 0.33 0.33];
q = [0.33;0.33;0.33];
% probability of states
y1 = [y0 y0+rho y0+2*rho];
y2 = [1;2;3];
y = y1 + y2;
c_fb = alpha.*a1.*y;
d_fb = (1-alpha).*a2.*y;
Q_fb = (1-alpha.*a1-(1-alpha).*a2).*y./price;
u_fb = sum(a1.*log(c_fb) + (1-a1).*log(Q_fb))./3;
eu = sum(u_fb)./3;
pvar = zeros(n,3);
pvar(:,2) = linspace(0.1,0.9,n)';
pvar(:,1) = (u_fb(2)-pvar(:,2).*u_fb(2)-(1-pvar(:,2)).*u_fb(3))./(u_fb(1)-u_fb(3));
pvar(:,3) = 1-pvar(:,2)-pvar(:,1);

%% Simulations
% First simulation: vary cost of hiding for agent 1
eu1 = zeros(n,6);
ev1 = zeros(n,6);
eh1 = zeros(n,6);
esp1 = zeros(n,6);
ems1 = zeros(n,5);
us1 = zeros(n,5);
vs1 = zeros(n,5);
Q1 = zeros(n,5);
parfor i = 1:n
    tic
    [eu1(i,:), ev1(i,:), eh1(i,:), esp1(i,:), ems1(i,:), us1(i,:), vs1(i,:), Q1(i,:)] = bilateral_hiding_public_good(a1,a2,price,deltavar(i),delta2,alpha,y0,rho,p,q);
    toc
end

%% First figure varies the cost of hiding
figure;
%figure(1);
subplot(2,5,1);
plot(deltavar,eu1(:,1),deltavar,eu1(:,2),deltavar,eu1(:,3),deltavar,eu1(:,4),deltavar,eu1(:,5));
title('Expected Utility of Agent 1');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Utility');

subplot(2,5,2);
plot(deltavar,ev1(:,1),deltavar,ev1(:,2),deltavar,ev1(:,3),deltavar,ev1(:,4),deltavar,ev1(:,5));
title('Expected Utility of Agent 2');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Utility');
legend('hide');

subplot(2,5,3);
plot(deltavar,eh1(:,1),deltavar,eh1(:,2),deltavar,eh1(:,3),deltavar,eh1(:,4),deltavar,eh1(:,5));
title('Expected Utility of Household');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Utility');
legend('hide');

subplot(2,5,4);
plot(deltavar,esp1(:,1),deltavar,esp1(:,2),deltavar,esp1(:,3),deltavar,esp1(:,4),deltavar,esp1(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Utility');
legend('hide');

subplot(2,5,6);
plot(deltavar,us1(:,1),deltavar,us1(:,2),deltavar,us1(:,3),deltavar,us1(:,4),deltavar,us1(:,5));
title('Share of Surplus of Agent 1');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Share');
legend('hide');

subplot(2,5,7);
plot(deltavar,vs1(:,1),deltavar,vs1(:,2),deltavar,vs1(:,3),deltavar,vs1(:,4),deltavar,vs1(:,5));
title('Share of Surplus of Agent 2');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Share');
legend('hide');

subplot(2,5,8);
plot(deltavar,ems1(:,1),deltavar,ems1(:,2),deltavar,ems1(:,3),deltavar,ems1(:,4),deltavar,ems1(:,5));
title('Marriage Surplus');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Utility');
legend('hide');

subplot(2,5,9);
plot(deltavar,Q1(:,1),deltavar,Q1(:,2),deltavar,Q1(:,3),deltavar,Q1(:,4),deltavar,Q1(:,5));
title('Public Good Consumption');
xlabel('(Inverse) Cost of Hiding \delta_1');
ylabel('Utility');
legend('hide');

subplot(2,5,5);
plot(0,0,0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.7 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_public_good_costofhiding.eps','epsc');

%% Second simulation: vary Pareto weight
eu2 = zeros(n,6);
ev2 = zeros(n,6);
eh2 = zeros(n,6);
esp2 = zeros(n,6);
ems2 = zeros(n,5);
us2 = zeros(n,5);
vs2 = zeros(n,5);
Q2 = zeros(n,5);
parfor i = 1:n
    tic
    [eu2(i,:), ev2(i,:), eh2(i,:), esp2(i,:), ems2(i,:), us2(i,:), vs2(i,:), Q2(i,:)] = bilateral_hiding_public_good(a1,a2,price,delta1,delta2,alphavar(i),y0,rho,p,q);
    toc 
end

%% Second figure varies Pareto weights
figure;
subplot(2,5,1);
plot(alphavar,eu2(:,1),alphavar,eu2(:,2),alphavar,eu2(:,3),alphavar,eu2(:,4),alphavar,eu2(:,5));
title('Expected Utility of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');

subplot(2,5,2);
plot(alphavar,ev2(:,1),alphavar,ev2(:,2),alphavar,ev2(:,3),alphavar,ev2(:,4),alphavar,ev2(:,5));
title('Expected Utility of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,5,3);
plot(alphavar,eh2(:,1),alphavar,eh2(:,2),alphavar,eh2(:,3),alphavar,eh2(:,4),alphavar,eh2(:,5));
title('Expected Utility of Household');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,5,4);
plot(alphavar,esp2(:,1),alphavar,esp2(:,2),alphavar,esp2(:,3),alphavar,esp2(:,4),alphavar,esp2(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,5,6);
plot(alphavar,us2(:,1),alphavar,us2(:,2),alphavar,us2(:,3),alphavar,us2(:,4),alphavar,us2(:,5));
title('Share of Surplus of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,5,7);
plot(alphavar,vs2(:,1),alphavar,vs2(:,2),alphavar,vs2(:,3),alphavar,vs2(:,4),alphavar,vs2(:,5));
title('Share of Surplus of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,5,8);
plot(alphavar,ems2(:,1),alphavar,ems2(:,2),alphavar,ems2(:,3),alphavar,ems2(:,4),alphavar,ems2(:,5));
title('Marriage Surplus');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,5,9);
plot(alphavar,Q2(:,1),alphavar,Q2(:,2),alphavar,Q2(:,3),alphavar,Q2(:,4),alphavar,Q2(:,5));
title('Public Good Consumption');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,5,5);
plot(0,0,0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.7 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_public_good_paretoweight.eps','epsc');



%% Third simulation: vary public good price
eu3 = zeros(n,6);
ev3 = zeros(n,6);
eh3 = zeros(n,6);
esp3 = zeros(n,6);
ems3 = zeros(n,5);
us3 = zeros(n,5);
vs3 = zeros(n,5);
Q3 = zeros(n,5);
parfor i = 1:n
    tic
    [eu3(i,:), ev3(i,:), eh3(i,:), esp3(i,:), ems3(i,:), us3(i,:), vs3(i,:), Q3(i,:)] = bilateral_hiding_public_good(a1,a2,pricevar(i),delta1,delta2,alpha,y0,rho,p,q);
    toc 
end

%% Third figure varies public good price
figure;

subplot(2,5,1);
plot(pricevar,eu3(:,1),pricevar,eu3(:,2),pricevar,eu3(:,3),pricevar,eu3(:,4),pricevar,eu3(:,5));
title('Expected Utility of Agent 1');
xlabel('Price of Public Good');
ylabel('Utility');

subplot(2,5,2);
plot(pricevar,ev3(:,1),pricevar,ev3(:,2),pricevar,ev3(:,3),pricevar,ev3(:,4),pricevar,ev3(:,5));
title('Expected Utility of Agent 2');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,5,3);
plot(pricevar,eh3(:,1),pricevar,eh3(:,2),pricevar,eh3(:,3),pricevar,eh3(:,4),pricevar,eh3(:,5));
title('Expected Utility of Household');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,5,4);
plot(pricevar,esp3(:,1),pricevar,esp3(:,2),pricevar,esp3(:,3),pricevar,esp3(:,4),pricevar,esp3(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,5,6);
plot(pricevar,us3(:,1),pricevar,us3(:,2),pricevar,us3(:,3),pricevar,us3(:,4),pricevar,us3(:,5));
title('Share of Surplus of Agent 1');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,5,7);
plot(pricevar,vs3(:,1),pricevar,vs3(:,2),pricevar,vs3(:,3),pricevar,vs3(:,4),pricevar,vs3(:,5));
title('Share of Surplus of Agent 2');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,5,8);
plot(pricevar,ems3(:,1),pricevar,ems3(:,2),pricevar,ems3(:,3),pricevar,ems3(:,4),pricevar,ems3(:,5));
title('Marriage Surplus');
xlabel('Price of Public Good');
ylabel('Utility');
legend('hide');

subplot(2,5,9);
plot(pricevar,Q3(:,1),pricevar,Q3(:,2),pricevar,Q3(:,3),pricevar,Q3(:,4),pricevar,Q3(:,5));
title('Public Good Consumption');
xlabel('Price of Public Good');
ylabel('Consumption');
legend('hide');

subplot(2,5,5);
plot(0,0,0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.7 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_public_good_price.eps','epsc');


%% Fourth simulation: vary preference of public good
eu4 = zeros(n,6);
ev4 = zeros(n,6);
eh4 = zeros(n,6);
esp4 = zeros(n,6);
ems4 = zeros(n,5);
us4 = zeros(n,5);
vs4 = zeros(n,5);
Q4 = zeros(n,5);
parfor i = 1:n
    tic
    [eu4(i,:), ev4(i,:), eh4(i,:), esp4(i,:), ems4(i,:), us4(i,:), vs4(i,:), Q4(i,:)] = bilateral_hiding_public_good(avar(i),a2,price,delta1,delta2,alpha,y0,rho,p,q);
    toc 
end

%% Fourth figure varies preference of public good
figure;

subplot(2,5,1);
plot(avar,eu4(:,1),avar,eu4(:,2),avar,eu4(:,3),avar,eu4(:,4),avar,eu4(:,5));
title('Expected Utility of Agent 1');
xlabel('Preference for Private Good of Agent 1');
ylabel('Utility');

subplot(2,5,2);
plot(avar,ev4(:,1),avar,ev4(:,2),avar,ev4(:,3),avar,ev4(:,4),avar,ev4(:,5));
title('Expected Utility of Agent 2');
xlabel('Preference for Private Good of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,3);
plot(avar,eh4(:,1),avar,eh4(:,2),avar,eh4(:,3),avar,eh4(:,4),avar,eh4(:,5));
title('Expected Utility of Household');
xlabel('Preference for Private Good of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,4);
plot(avar,esp4(:,1),avar,esp4(:,2),avar,esp4(:,3),avar,esp4(:,4),avar,esp4(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Preference for Private Good of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,6);
plot(avar,us4(:,1),avar,us4(:,2),avar,us4(:,3),avar,us4(:,4),avar,us4(:,5));
title('Share of Surplus of Agent 1');
xlabel('Preference for Private Good of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,7);
plot(avar,vs4(:,1),avar,vs4(:,2),avar,vs4(:,3),avar,vs4(:,4),avar,vs4(:,5));
title('Share of Surplus of Agent 2');
xlabel('Preference for Private Good of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,8);
plot(avar,ems4(:,1),avar,ems4(:,2),avar,ems4(:,3),avar,ems4(:,4),avar,ems4(:,5));
title('Marriage Surplus');
xlabel('Preference for Private Good of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,9);
plot(avar,Q4(:,1),avar,Q4(:,2),avar,Q4(:,3),avar,Q4(:,4),avar,Q4(:,5));
title('Public Good Consumption');
xlabel('Preference for Private Good of Agent 1');
ylabel('Consumption');
legend('hide');

subplot(2,5,5);
plot(0,0,0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.7 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_public_good_preference.eps','epsc');

%% Fifth simulation: vary mean income of agent 1
eu5 = zeros(n,6);
ev5 = zeros(n,6);
eh5 = zeros(n,6);
esp5 = zeros(n,6);
ems5 = zeros(n,5);
us5 = zeros(n,5);
vs5 = zeros(n,5);
Q5 = zeros(n,5);
parfor i = 1:n
    tic
    [eu5(i,:), ev5(i,:), eh5(i,:), esp5(i,:), ems5(i,:), us5(i,:), vs5(i,:), Q5(i,:)] = bilateral_hiding_public_good(a1,a2,price,delta1,delta2,alpha,yvar(i),rho,p,q);
    toc 
end

%% Fifth figure varies mean income of agent 1
figure;

subplot(2,5,1);
plot(yvar,eu5(:,1),yvar,eu5(:,2),yvar,eu5(:,3),yvar,eu5(:,4),yvar,eu5(:,5));
title('Expected Utility of Agent 1');
xlabel('Income of Agent 1');
ylabel('Utility');

subplot(2,5,2);
plot(yvar,ev5(:,1),yvar,ev5(:,2),yvar,ev5(:,3),yvar,ev5(:,4),yvar,ev5(:,5));
title('Expected Utility of Agent 2');
xlabel('Preference for Private Good of Agent 1');
xlabel('Income of Agent 1');
legend('hide');

subplot(2,5,3);
plot(yvar,eh5(:,1),yvar,eh5(:,2),yvar,eh5(:,3),yvar,eh5(:,4),yvar,eh5(:,5));
title('Expected Utility of Household');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,4);
plot(yvar,esp5(:,1),yvar,esp5(:,2),yvar,esp5(:,3),yvar,esp5(:,4),yvar,esp5(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,6);
plot(yvar,us5(:,1),yvar,us5(:,2),yvar,us5(:,3),yvar,us5(:,4),yvar,us5(:,5));
title('Share of Surplus of Agent 1');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,7);
plot(yvar,vs5(:,1),yvar,vs5(:,2),yvar,vs5(:,3),yvar,vs5(:,4),yvar,vs5(:,5));
title('Share of Surplus of Agent 2');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,8);
plot(yvar,ems5(:,1),yvar,ems5(:,2),yvar,ems5(:,3),yvar,ems5(:,4),yvar,ems5(:,5));
title('Marriage Surplus');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,9);
plot(yvar,Q5(:,1),yvar,Q5(:,2),yvar,Q5(:,3),yvar,Q5(:,4),yvar,Q5(:,5));
title('Public Good Consumption');
xlabel('Income of Agent 1');
ylabel('Consumption');
legend('hide');

subplot(2,5,5);
plot(0,0,0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.7 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_public_good_income.eps','epsc');

%% Sixth simulation: vary income variability of agent 1
eu6 = zeros(n,6);
ev6 = zeros(n,6);
eh6 = zeros(n,6);
esp6 = zeros(n,6);
ems6 = zeros(n,5);
us6 = zeros(n,5);
vs6 = zeros(n,5);
Q6 = zeros(n,5);
parfor i = 1:n
    tic
    [eu6(i,:), ev6(i,:), eh6(i,:), esp6(i,:), ems6(i,:), us6(i,:), vs6(i,:), Q6(i,:)] = bilateral_hiding_public_good(a1,a2,price,delta1,delta2,alpha,ycons(i),rhovar(i),p,q);
    toc 
end

%% Sixth figure varies income variability of agent 1
figure;

subplot(2,5,1);
plot(rhovar,eu6(:,1),rhovar,eu6(:,2),rhovar,eu6(:,3),rhovar,eu6(:,4),rhovar,eu6(:,5));
title('Expected Utility of Agent 1');
xlabel('Income Variability of Agent 1');
ylabel('Utility');

subplot(2,5,2);
plot(rhovar,ev6(:,1),rhovar,ev6(:,2),rhovar,ev6(:,3),rhovar,ev6(:,4),rhovar,ev6(:,5));
title('Expected Utility of Agent 2');
xlabel('Income Variability of Agent 1');
legend('hide');

subplot(2,5,3);
plot(rhovar,eh6(:,1),rhovar,eh6(:,2),rhovar,eh6(:,3),rhovar,eh6(:,4),rhovar,eh6(:,5));
title('Expected Utility of Household');
xlabel('Income Variability of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,4);
plot(rhovar,esp6(:,1),rhovar,esp6(:,2),rhovar,esp6(:,3),rhovar,esp6(:,4),rhovar,esp6(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Income Variability of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,6);
plot(rhovar,us6(:,1),rhovar,us6(:,2),rhovar,us6(:,3),rhovar,us6(:,4),rhovar,us6(:,5));
title('Share of Surplus of Agent 1');
xlabel('Income Variability of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,7);
plot(rhovar,vs6(:,1),rhovar,vs6(:,2),rhovar,vs6(:,3),rhovar,vs6(:,4),rhovar,vs6(:,5));
title('Share of Surplus of Agent 2');
xlabel('Income Variability of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,8);
plot(rhovar,ems6(:,1),rhovar,ems6(:,2),rhovar,ems6(:,3),rhovar,ems6(:,4),rhovar,ems6(:,5));
title('Marriage Surplus');
xlabel('Income Variability of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,9);
plot(rhovar,Q6(:,1),rhovar,Q6(:,2),rhovar,Q6(:,3),rhovar,Q6(:,4),rhovar,Q6(:,5));
title('Public Good Consumption');
xlabel('Income Variability of Agent 1');
ylabel('Consumption');
legend('hide');

subplot(2,5,5);
plot(0,0,0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.7 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_public_good_variability.eps','epsc');

%% Seventh simulation: vary probability of states for agent 1
eu7 = zeros(n,6);
ev7 = zeros(n,6);
eh7 = zeros(n,6);
esp7 = zeros(n,6);
ems7 = zeros(n,5);
us7 = zeros(n,5);
vs7 = zeros(n,5);
Q7 = zeros(n,5);
parfor i = 1:n
    tic
    [eu7(i,:), ev7(i,:), eh7(i,:), esp7(i,:), ems7(i,:), us7(i,:), vs7(i,:), Q7(i,:)] = bilateral_hiding_public_good(a1,a2,price,delta1,delta2,alpha,y0,rho,pvar(i,:),q);
    toc 
end

%% Seventh figure varies probability of states for agent 1
figure;

subplot(2,5,1);
plot(pvar(:,2),eu7(:,1),pvar(:,2),eu7(:,2),pvar(:,2),eu7(:,3),pvar(:,2),eu7(:,4),pvar(:,2),eu7(:,5));
title('Expected Utility of Agent 1');
xlabel('Probability of State 2 for Agent 1');
ylabel('Utility');

subplot(2,5,2);
plot(pvar(:,2),ev7(:,1),pvar(:,2),ev7(:,2),pvar(:,2),ev7(:,3),pvar(:,2),ev7(:,4),pvar(:,2),ev7(:,5));
title('Expected Utility of Agent 2');
xlabel('Probability of State 2 for Agent 1');
legend('hide');

subplot(2,5,3);
plot(pvar(:,2),eh7(:,1),pvar(:,2),eh7(:,2),pvar(:,2),eh7(:,3),pvar(:,2),eh7(:,4),pvar(:,2),eh7(:,5));
title('Expected Utility of Household');
xlabel('Probability of State 2 for Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,4);
plot(pvar(:,2),esp7(:,1),pvar(:,2),esp7(:,2),pvar(:,2),esp7(:,3),pvar(:,2),esp7(:,4),pvar(:,2),esp7(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Probability of State 2 for Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,6);
plot(pvar(:,2),us7(:,1),pvar(:,2),us7(:,2),pvar(:,2),us7(:,3),pvar(:,2),us7(:,4),pvar(:,2),us7(:,5));
title('Share of Surplus of Agent 1');
xlabel('Probability of State 2 for Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,7);
plot(pvar(:,2),vs7(:,1),pvar(:,2),vs7(:,2),pvar(:,2),vs7(:,3),pvar(:,2),vs7(:,4),pvar(:,2),vs7(:,5));
title('Share of Surplus of Agent 2');
xlabel('Probability of State 2 for Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,8);
plot(pvar(:,2),ems7(:,1),pvar(:,2),ems7(:,2),pvar(:,2),ems7(:,3),pvar(:,2),ems7(:,4),pvar(:,2),ems7(:,5));
title('Marriage Surplus');
xlabel('Probability of State 2 for Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,5,9);
plot(pvar(:,2),Q7(:,1),pvar(:,2),Q7(:,2),pvar(:,2),Q7(:,3),pvar(:,2),Q7(:,4),pvar(:,2),Q7(:,5));
title('Public Good Consumption');
xlabel('Probability of State 2 for Agent 1');
ylabel('Consumption');
legend('hide');

subplot(2,5,5);
plot(0,0,0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.7 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_public_good_probability.eps','epsc');