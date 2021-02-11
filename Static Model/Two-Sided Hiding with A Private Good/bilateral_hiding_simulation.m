% Bilateral Hiding Simulation
% Sally Zhang
% April 23, 2020
% Simulation of a two agent static model where both agents can hide. Both
% agents have three states of the world and the realizations of the two are
% independent.
% April 23: add social planner utility

clear all; close all; clc;

%% Set Up
% Parameters
% number of simulations
n = 30;
% delta1, delta2 [0,1]: cost of hiding. delta=1 means there is no cost of hiding.
% delta=0 means cost of hiding is prohibitive.
delta0=0.9;
delta = linspace(0,1,n);
% alpha [0,1] is the Pareto weight of agent 1.
alpha0=0.3;
alpha = linspace(0.1,0.9,n);
% y is the income
y0 = 1;
yvar = linspace(0.5,2,n);
rho0 = 1;
rho = linspace(0.1,1.9,n);
% Generate an income process with increasing variance but constant mean
ycons = 2 - rho;
% p is the probability of states
p = [0.33 0.33 0.33];
q = [0.33;0.33;0.33];
% probability of states
pvar = zeros(n,3);
pvar(:,2) = linspace(0.1,0.8,n)';
[u, eu, ~, eh, ~] = bilateral_hiding_log(delta0, delta0, alpha0, y0, rho0, p, q); 
qeu = q'*u(:,:,1);
pvar(:,1) = (eu(1) - pvar(:,2).*qeu(2))./(qeu(1)+qeu(3));
pvar(:,3) = pvar(:,1);

%% Simulations
% First simulation: vary cost of hiding
eu1 = zeros(n,5);
ev1 = zeros(n,5);
eh1 = zeros(n,5);
esp1 = zeros(n,5);
ems1 = zeros(n,5);
us1 = zeros(n,5);
vs1 = zeros(n,5);
for i = 1:n
    [~, eu1(i,:), ev1(i,:), eh1(i,:), esp1(i,:), ems1(i,:), us1(i,:), vs1(i,:)] = bilateral_hiding_log_utility(delta(i), delta0, alpha0, y0, rho0, p, q); 
end

% Second simulation: vary Pareto weights
eu2 = zeros(n,5);
ev2 = zeros(n,5);
eh2 = zeros(n,5);
esp2 = zeros(n,5);
ems2 = zeros(n,5);
us2 = zeros(n,5);
vs2 = zeros(n,5);
for i = 1:n
    [~, eu2(i,:), ev2(i,:), eh2(i,:), esp2(i,:), ems2(i,:), us2(i,:), vs2(i,:)] = bilateral_hiding_log_utility(delta0, delta0, alpha(i), y0, rho0, p, q);
end

% Third simulation: vary probability of state 2 for agent 1
eu3 = zeros(n,5);
ev3 = zeros(n,5);
eh3 = zeros(n,5);
esp3 = zeros(n,5);
ems3 = zeros(n,5);
us3 = zeros(n,5);
vs3 = zeros(n,5);
for i = 1:n
    [~, eu3(i,:), ev3(i,:), eh3(i,:), esp3(i,:), ems3(i,:), us3(i,:), vs3(i,:)] = bilateral_hiding_log_utility(delta0, delta0, alpha0, y0, rho0, pvar(i,:), q);
end

% Fourth simulation: vary income
eu4 = zeros(n,5);
ev4 = zeros(n,5);
eh4 = zeros(n,5);
esp4 = zeros(n,5);
ems4 = zeros(n,5);
us4 = zeros(n,5);
vs4 = zeros(n,5);
for i = 1:n
    [~, eu4(i,:), ev4(i,:), eh4(i,:), esp4(i,:), ems4(i,:), us4(i,:), vs4(i,:)]= bilateral_hiding_log_utility(delta0, delta0, alpha0, yvar(i), rho0, p, q);
end

% Fifth simulation: vary income gaps
eu5 = zeros(n,5);
ev5 = zeros(n,5);
eh5 = zeros(n,5);
esp5 = zeros(n,5);
ems5 = zeros(n,5);
us5 = zeros(n,5);
vs5 = zeros(n,5);
for i = 1:n
    [~, eu5(i,:), ev5(i,:), eh5(i,:), esp5(i,:), ems5(i,:), us5(i,:), vs5(i,:)]= bilateral_hiding_log_utility(delta0, delta0, alpha0, ycons(i), rho(i), p, q);
end

%% Plot results
% Format axes
ymin = min([min(min(eu1)) min(min(ev1)) min(min(eh1))]);
ymax = max([max(max(eu1)) max(max(ev1)) max(max(eh1))]);

% First figure varies the cost of hiding
figure(1);
subplot(2,4,1);
plot(delta,eu1(:,1),delta,eu1(:,2),delta,eu1(:,3),delta,eu1(:,4),delta,eu1(:,5));
title('Expected Utility of Agent 1');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(delta,ev1(:,1),delta,ev1(:,2),delta,ev1(:,3),delta,ev1(:,4),delta,ev1(:,5));
title('Expected Utility of Agent 2');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(delta,eh1(:,1),delta,eh1(:,2),delta,eh1(:,3),delta,eh1(:,4),delta,eh1(:,5));
title('Expected Utility of Household');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(delta,esp1(:,1),delta,esp1(:,2),delta,esp1(:,3),delta,esp1(:,4),delta,esp1(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(delta,us1(:,1),delta,us1(:,2),delta,us1(:,3),delta,us1(:,4),delta,us1(:,5));
title('Marriage Surplus of Agent 1');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,6);
plot(delta,vs1(:,1),delta,vs1(:,2),delta,vs1(:,3),delta,vs1(:,4),delta,vs1(:,5));
title('Marriage Surplus of Agent 2');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,7);
plot(delta,ems1(:,1),delta,ems1(:,2),delta,ems1(:,3),delta,ems1(:,4),delta,ems1(:,5));
title('Marriage Surplus');
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.3 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_hiding_costofhiding.eps','epsc');


%% Second figure varies agent 1's Pareto weight
% Format axes
ymin = min([min(min(eu2)) min(min(ev2)) min(min(eh2))]);
ymax = max([max(max(eu2)) max(max(ev2)) max(max(eh2))]);

figure(2);
subplot(2,4,1);
plot(alpha,eu2(:,1),alpha,eu2(:,2),alpha,eu2(:,3),alpha,eu2(:,4),alpha,eu2(:,4));
title('Expected Utility of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
ylim([ymin ymax]);
legend('hide');

subplot(2,4,2);
plot(alpha,ev2(:,1),alpha,ev2(:,2),alpha,ev2(:,3),alpha,ev2(:,4),alpha,ev2(:,5));
title('Expected Utility of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
ylim([ymin ymax]);
legend('hide');

subplot(2,4,3);
plot(alpha,eh2(:,1),alpha,eh2(:,2),alpha,eh2(:,3),alpha,eh2(:,4),alpha,eh2(:,5))
title('Expected Utility of Household');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(alpha,esp2(:,1),alpha,esp2(:,2),alpha,esp2(:,3),alpha,esp2(:,4),alpha,esp2(:,5))
title('Expected Utility (Equal Weighted)');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(alpha,us2(:,1),alpha,us2(:,2),alpha,us2(:,3),alpha,us2(:,4),alpha,us2(:,5));
title('Marriage Surplus of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,6);
plot(alpha,vs2(:,1),alpha,vs2(:,2),alpha,vs2(:,3),alpha,vs2(:,4),alpha,vs2(:,5));
title('Marriage Surplus of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,7);
plot(alpha,ems2(:,1),alpha,ems2(:,2),alpha,ems2(:,3),alpha,ems2(:,4),alpha,ems2(:,5));
title('Marriage Surplus');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.3 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_hiding_paretoweight.eps','epsc');

% Format axes
ymin = min([min(min(eu3)) min(min(ev3)) min(min(eh3))]);
ymax = max([max(max(eu3)) max(max(ev3)) max(max(eh3))]);

% Third figure varies the likelihood of state 2
figure(3);
subplot(2,4,1);
plot(pvar(:,2),eu3(:,1),pvar(:,2),eu3(:,2),pvar(:,2),eu3(:,3),pvar(:,2),eu3(:,4),pvar(:,2),eu3(:,5));
title('Expected Utility of Agent 1');
xlabel('Probability of State 2');
ylabel('Utility');
xlim([0.1 0.8]);
legend('hide');

subplot(2,4,2);
plot(pvar(:,2),ev3(:,1),pvar(:,2),ev3(:,2),pvar(:,2),ev3(:,3),pvar(:,2),ev3(:,4),pvar(:,2),ev3(:,5));
title('Expected Utility of Agent 2');
xlabel('Probability of State 2');
ylabel('Utility');
xlim([0.1 0.8]);
legend('hide');

subplot(2,4,3);
plot(pvar(:,2),eh3(:,1),pvar(:,2),eh3(:,2),pvar(:,2),eh3(:,3),pvar(:,2),eh3(:,4),pvar(:,2),eh3(:,5));
title('Expected Utility of Household');
xlabel('Probability of State 2');
ylabel('Utility');
xlim([0.1 0.8]);
legend('hide');

subplot(2,4,4);
plot(pvar(:,2),esp3(:,1),pvar(:,2),esp3(:,2),pvar(:,2),esp3(:,3),pvar(:,2),esp3(:,4),pvar(:,2),esp3(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Probability of State 2');
ylabel('Utility');
xlim([0.1 0.8]);
legend('hide');

subplot(2,4,5);
plot(pvar(:,2),us3(:,1),pvar(:,2),us3(:,2),pvar(:,2),us3(:,3),pvar(:,2),us3(:,4),pvar(:,2),us3(:,5));
title('Marriage Surplus of Agent 1');
xlabel('Probability of State 2');
ylabel('Utility');
xlim([0.1 0.8]);
legend('hide');

subplot(2,4,6);
plot(pvar(:,2),vs3(:,1),pvar(:,2),vs3(:,2),pvar(:,2),vs3(:,3),pvar(:,2),vs3(:,4),pvar(:,2),vs3(:,5));
title('Marriage Surplus of Agent 2');
xlabel('Probability of State 2');
ylabel('Utility');
xlim([0.1 0.8]);
legend('hide');

subplot(2,4,7);
plot(pvar(:,2),ems3(:,1),pvar(:,2),ems3(:,2),pvar(:,2),ems3(:,3),pvar(:,2),ems3(:,4),pvar(:,2),ems3(:,5));
title('Marriage Surplus');
xlabel('Probability of State 2');
ylabel('Utility');
xlim([0.1 0.8]);
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.3 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_hiding_probability.eps','epsc');

%% Fourth figure varies income
% Format axes
ymin = min([min(min(eu4)) min(min(ev4)) min(min(eh4))]);
ymax = max([max(max(eu4)) max(max(ev4)) max(max(eh4))]);

figure(4);
subplot(2,4,1);
plot(yvar,eu4(:,1),yvar,eu4(:,2),yvar,eu4(:,3),yvar,eu4(:,4),yvar,eu4(:,5));
title('Expected Utility of Agent 1');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(yvar,ev4(:,1),yvar,ev4(:,2),yvar,ev4(:,3),yvar,ev4(:,4),yvar,ev4(:,5));
title('Expected Utility of Agent 2');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(yvar,eh4(:,1),yvar,eh4(:,2),yvar,eh4(:,3),yvar,eh4(:,4),yvar,eh4(:,5));
title('Expected Utility of Household');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(yvar,esp4(:,1),yvar,esp4(:,2),yvar,esp4(:,3),yvar,esp4(:,4),yvar,esp4(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(yvar,us4(:,1),yvar,us4(:,2),yvar,us4(:,3),yvar,us4(:,4),yvar,us4(:,5));
title('Marriage Surplus of Agent 1');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,4,6);
plot(yvar,vs4(:,1),yvar,vs4(:,2),yvar,vs4(:,3),yvar,vs4(:,4),yvar,vs4(:,5));
title('Marriage Surplus of Agent 2');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,4,7);
plot(yvar,ems4(:,1),yvar,ems4(:,2),yvar,ems4(:,3),yvar,ems4(:,4),yvar,ems4(:,5));
title('Marriage Surplus');
xlabel('Income of Agent 1');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.3 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_hiding_income.eps','epsc');

%% Fifth figure varies income variability
% Format axes
ymin = min([min(min(eu5)) min(min(ev5)) min(min(eh5))]);
ymax = max([max(max(eu5)) max(max(ev5)) max(max(eh5))]);

figure(5);
subplot(2,4,1);
plot(rho,eu5(:,1),rho,eu5(:,2),rho,eu5(:,3),rho,eu5(:,4),rho,eu5(:,5));
title('Expected Utility of Agent 1');
xlabel('Income Variability of Agent 1 \rho');
ylabel('Utility');
legend('hide');

subplot(2,4,2);
plot(rho,ev5(:,1),rho,ev5(:,2),rho,ev5(:,3),rho,ev5(:,4),rho,ev5(:,5));
title('Expected Utility of Agent 2');
xlabel('Income Variability of Agent 1 \rho');
ylabel('Utility');
legend('hide');

subplot(2,4,3);
plot(rho,eh5(:,1),rho,eh5(:,2),rho,eh5(:,3),rho,eh5(:,4),rho,eh5(:,5));
title('Expected Utility of Household');
xlabel('Income Variability of Agent 1 \rho');
ylabel('Utility');
legend('hide');

subplot(2,4,4);
plot(rho,esp5(:,1),rho,esp5(:,2),rho,esp5(:,3),rho,esp5(:,4),rho,esp5(:,5));
title('Expected Utility (Equal Weighted)');
xlabel('Income Variability of Agent 1 \rho');
ylabel('Utility');
legend('hide');

subplot(2,4,5);
plot(rho,us5(:,1),rho,us5(:,2),rho,us5(:,3),rho,us5(:,4),rho,us5(:,5));
title('Marriage Surplus of Agent 1');
xlabel('Income Variability of Agent 1 \rho');
ylabel('Utility');
legend('hide');

subplot(2,4,6);
plot(rho,vs5(:,1),rho,vs5(:,2),rho,vs5(:,3),rho,vs5(:,4),rho,vs5(:,5));
title('Marriage Surplus of Agent 2');
xlabel('Income Variability of Agent 1 \rho');
ylabel('Utility');
legend('hide');

subplot(2,4,7);
plot(rho,ems5(:,1),rho,ems5(:,2),rho,ems5(:,3),rho,ems5(:,4),rho,ems5(:,5));
title('Marriage Surplus');
xlabel('Income Variability of Agent 1 \rho');
ylabel('Utility');
legend('hide');

subplot(2,4,8);
plot(0,0,0,0,0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Agent 1 Hides','Agent 2 Hides','Both Hides','Position',[0.75 0.3 0.1 0.1]);

set(gcf, 'Position',  [100, 100, 1300, 500])
saveas(gcf,'C:\Users\tz012\Dropbox\Projects and Ideas\Hidden Income\Model\Output\bilateral_hiding_variability.eps','epsc');
