% Unilateral Hiding Simulation
% Sally Zhang
% April 14, 2020
% Simulation of a two agent static model where only one agent can hide.

clear all; close all; clc;

%% Set Up
% Parameters
% delta [0,1] is cost of hiding. delta=1 means there is no cost of hiding.
% delta=0 means cost of hiding is prohibitive.
delta0=0.9;
delta = linspace(0,1,10);
% alpha [0,1] is the Pareto weight of agent 1.
alpha0=0.5;
alpha = linspace(0.1,0.9,10);
% y is the income
y0 = 3;
y = [y0;y0+1;y0+2];
yvar = linspace(1,5,10);
yvarall = [yvar;yvar+1;yvar+2];
% p is the probability of states
p0 = [0.33 0.33 0.33];
ey = p0*y;
eyvar = p0*yvarall;
% baseline values
[u, eu, v, ev, h, eh] = unilateral_hiding2(delta0, alpha0, y0, p0); 
% probability of states
p = zeros(10,3);
p(:,2) = linspace(0.1,0.9,10)';
p(:,1) = (eu(1) - p(:,2).*u(2,1))./(u(1,1)+u(3,1));
p(:,3) = p(:,1);

%% Simulations
% First simulation: vary cost of hiding
eu1 = zeros(10,3);
ev1 = zeros(10,3);
eh1 = zeros(10,3);
for i = 1:10
    [~, eu1(i,:), ~, ev1(i,:), ~, eh1(i,:)]=unilateral_hiding(delta(i), alpha0, y0, p0);
end
% Second simulation: vary Pareto weights
eu2 = zeros(10,3);
ev2 = zeros(10,3);
eh2 = zeros(10,3);
for i = 1:10
    [~, eu2(i,:), ~, ev2(i,:), ~, eh2(i,:)]=unilateral_hiding(delta0, alpha(i), y0, p0);
end
% Third simulation: vary probability of state 2
eu3 = zeros(10,3);
ev3 = zeros(10,3);
eh3 = zeros(10,3);
for i = 1:10
    [~, eu3(i,:), ~, ev3(i,:), ~, eh3(i,:)]=unilateral_hiding(delta0, alpha0, y0, p(i,:));
end
% Fourth simulation: vary income
eu4 = zeros(10,3);
ev4 = zeros(10,3);
eh4 = zeros(10,3);
for i = 1:10
    [~, eu4(i,:), ~, ev4(i,:), ~, eh4(i,:)]=unilateral_hiding(delta0, alpha0, yvar(i), p0);
end

%% Plot results
% Format axes
ymin = min([min(min(eu1)) min(min(ev1)) min(min(eh1))]);
ymax = max([max(max(eu1)) max(max(ev1)) max(max(eh1))]);

% First figure varies the cost of hiding
figure(1);
subplot(1,4,1);
plot(delta,eu1(:,1),delta,eu1(:,2),delta,eu1(:,3))
title('Expected Utility of Agent 1');
ylim([ymin ymax]);
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(1,4,2);
plot(delta,ev1(:,1),delta,ev1(:,2),delta,ev1(:,3))
title('Expected Utility of Agent 2');
ylim([ymin ymax]);
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(1,4,3);
plot(delta,eh1(:,1),delta,eh1(:,2),delta,eh1(:,3))
title('Expected Utility of Household');
ylim([ymin ymax]);
xlabel('(Inverse) Cost of Hiding \delta');
ylabel('Utility');
legend('hide');

subplot(1,4,4);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.5 0.1 0.1]);

% Format axes
ymin = min([min(min(eu2)) min(min(ev2)) min(min(eh2))]);
ymax = max([max(max(eu2)) max(max(ev2)) max(max(eh2))]);

% Second figure varies agent 1's Pareto weight
figure(2);
subplot(1,4,1);
plot(alpha,eu2(:,1),alpha,eu2(:,2),alpha,eu2(:,3))
title('Expected Utility of Agent 1');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
ylim([ymin ymax]);
xlim([0.1 0.9]);
legend('hide');

subplot(1,4,2);
plot(alpha,ev2(:,1),alpha,ev2(:,2),alpha,ev2(:,3))
title('Expected Utility of Agent 2');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
ylim([ymin ymax]);
xlim([0.1 0.9]);
legend('hide');

subplot(1,4,3);
plot(alpha,eh2(:,1),alpha,eh2(:,2),alpha,eh2(:,3))
title('Expected Utility of Household');
xlabel('Pareto Weight of Agent 1 \alpha');
ylabel('Utility');
ylim([ymin ymax]);
xlim([0.1 0.9]);
legend('hide');

subplot(1,4,4);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.5 0.1 0.1]);

% Format axes
ymin = min([min(min(eu3)) min(min(ev3)) min(min(eh3))]);
ymax = max([max(max(eu3)) max(max(ev3)) max(max(eh3))]);

% Third figure varies the likelihood of state 2
figure(3);
subplot(1,4,1);
plot(p(:,2),eu3(:,1),p(:,2),eu3(:,2),p(:,2),eu3(:,3));
title('Expected Utility of Agent 1');
xlabel('Probability of State 2');
ylabel('Utility');
ylim([ymin ymax]);
xlim([0.1 0.9]);
legend('hide');

subplot(1,4,2);
plot(p(:,2),ev3(:,1),p(:,2),ev3(:,2),p(:,2),ev3(:,3));
title('Expected Utility of Agent 2');
xlabel('Probability of State 2');
ylabel('Utility');
ylim([ymin ymax]);
xlim([0.1 0.9]);
legend('hide');

subplot(1,4,3);
plot(p(:,2),eh3(:,1),p(:,2),eh3(:,2),p(:,2),eh3(:,3));
title('Expected Utility of Household');
xlabel('Probability of State 2');
ylabel('Utility');
ylim([ymin ymax]);
xlim([0.1 0.9]);
legend('hide');

subplot(1,4,4);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.5 0.1 0.1]);

% Format axes
ymin = min([min(min(eu4)) min(min(ev4)) min(min(eh4))]);
ymax = max([max(max(eu4)) max(max(ev4)) max(max(eh4))]);

% Fourth figure varies income
figure(4);
subplot(1,4,1);
plot(eyvar,eu4(:,1),eyvar,eu4(:,2),eyvar,eu4(:,3));
title('Expected Utility of Agent 1');
xlabel('Expected Household Income');
ylabel('Utility');
ylim([ymin ymax]);
xlim([2 6]);
legend('hide');

subplot(1,4,2);
plot(eyvar,ev4(:,1),eyvar,ev4(:,2),eyvar,ev4(:,3));
title('Expected Utility of Agent 2');
xlabel('Expected Household Income');
ylabel('Utility');
ylim([ymin ymax]);
xlim([2 6]);
legend('hide');

subplot(1,4,3);
plot(eyvar,eh4(:,1),eyvar,eh4(:,2),eyvar,eh4(:,3));
title('Expected Utility of Household');
xlabel('Expected Household Income');
ylabel('Utility');
ylim([ymin ymax]);
xlim([2 6]);
legend('hide');

subplot(1,4,4);
plot(0,0,0,0,0,0);
axis off;
legend('First-best','Honest','Hiding','Position',[0.75 0.5 0.1 0.1]);