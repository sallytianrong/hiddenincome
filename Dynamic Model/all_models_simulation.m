% Simulation of first-best, incentive compatible, and hiding equilibrium
% This code uses a common set of parameters, solve for both incentive
% compative and hiding equilibria using functions, and use those outputs
% for a simulation.

clear all; close all; clc;

%% Parameters
% Preference of private good versus public good
a1 = 0.3; a2 = 0.7;

% Two income processes: uniform discrete
y1min = 1; y1max = 5;
y2min = 1; y2max = 5;

% Pareto weight
alpha = 0.5;

% price
price = 1;

% Time discounting
beta = 0.94;

% discretize income space
n = 2;
income1 = linspace(y1min,y1max,n);
income2 = linspace(y2min,y2max,n);

% discretize state space for IC
ns = 25; nw = 32;

% utility function
u1 = @(x,Q) a1.*log(x) + (1-a1).*log(Q);
u2 = @(x,Q) a2.*log(x) + (1-a2).*log(Q);

% Autarky value
% autarky utility
uaut1 = @(y) a1.*log(a1.*y)+(1-a1).*log((1-a1).*y);
uaut2 = @(y) a2.*log(a2.*y)+(1-a2).*log((1-a2).*y);

% Cost of hiding
delta1 = 0.95; delta2 = 0.95;
deltavar = linspace(0.1,0.99,10);

%% Solve incentive compatible
[w0, w_feasible, P, nfeas, X_all] = IC_twoside_whiding(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,price,n,ns,nw,delta1,delta2);

% save workspace because this takes a long time to run! it will be easier
% if I don't need to run it every time
%save ('ic.mat');

%% Hiding

% load workspace
%load ('ic.mat');

% Hiding model solves much faster so can allow more nodes 
n_hiding = 100;
income1_hiding = linspace(y1min,y1max,n_hiding);
income2_hiding = linspace(y2min,y2max,n_hiding);

% a mapping from sparse to dense grid
map = zeros(1,n);
for i = 1:n
    [~, map(i)] = min(abs(income1(i)-income1_hiding));
end

% hiding
[z1max_bl, z2max_bl] = infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,delta1,delta2,price,n_hiding);

%% Simulation
rng(8);

% simulation time periods
T = 100;

% generate alternate income path
ind_path1 = repmat([1 2],1,T/2);
ind_path2 = repmat([2 1],1,T/2);

% Simulate income processes
%ind_path1 = randi([1 n],T,1);
%ind_path2 = randi([1 n],T,1);
ind_path1_hiding = map(ind_path1);
ind_path2_hiding = map(ind_path2);
y1_path = income1(ind_path1);
y2_path = income2(ind_path2);

%% Autarky
% autarky consumption path
c1_aut_path = a1.*y1_path;
c2_aut_path = a2.*y2_path;
Q_aut_path = (1-a1).*y1_path + (1-a2).*y2_path;
% autarky utility stream
aut1_path = uaut1(y1_path);
aut2_path = uaut2(y2_path);
% expected utility at time 0
eu1_aut = mean(uaut1(income1))./(1-beta);
eu2_aut = mean(uaut2(income2))./(1-beta);

%% First best
% first best consumption path
gamma = (1-a1).*alpha + (1-a2).*(1-alpha);
c1_fb_path = a1.*alpha.*(y1_path + y2_path);
c2_fb_path = a2.*(1-alpha).*(y1_path + y2_path);
Q_fb_path = gamma.*(y1_path + y2_path);
% first best utility path
u1_fb_path = u1(c1_fb_path,Q_fb_path);
u2_fb_path = u2(c2_fb_path,Q_fb_path);
surplus_fb_path = a1.*log(c1_fb_path) + (1-a1).*log(Q_fb_path) + a2.*log(c2_fb_path) + (1-a2).*log(Q_fb_path) - aut1_path - aut2_path;
% expected utility at time 0
eu1_fb = mean(u1_fb_path)./(1-beta);
eu2_fb = mean(u2_fb_path)./(1-beta);
eh_fb = alpha*eu1_fb + (1-alpha)*eu2_fb;
eq_fb = 0.5.*eu1_fb + 0.5.*eu2_fb;

%% simulate reported income stream
z1_path = zeros(1,T);
z2_path = zeros(1,T);
for t = 1:T
    z1_path(t) = z1max_bl(ind_path1_hiding(t),ind_path2_hiding(t));
    z2_path(t) = z2max_bl(ind_path1_hiding(t),ind_path2_hiding(t));
end

% simulate consumption and utility stream
c1_hide_path = alpha.*a1.*(z1_path+z2_path) + delta1.*(y1_path-z1_path);
c2_hide_path = (1-alpha).*a2.*(z1_path+z2_path) + delta2.*(y2_path-z2_path);
Q_hide_path = gamma.*(z1_path+z2_path);
u1_hide_path = u1(c1_hide_path,Q_hide_path);
u2_hide_path = u2(c2_hide_path,Q_hide_path);

% simulate probabilities of detection
prob1_path = ((y1_path-z1_path)./y1_path).^2;
prob2_path = ((y2_path-z2_path)./y2_path).^2;

% draw random numbers to see whether punishment state realizes
rand1_path = rand(1,T);
punishment1_path = (rand1_path<prob1_path);
rand2_path = rand(1,T);
punishment2_path = (rand2_path<prob2_path);
% first punishment state
first_punishment1 = find(punishment1_path,1);
first_punishment2 = find(punishment2_path,1);
first_punishment = min(first_punishment1,first_punishment2);

% all time periods after the first punishment state is autarky
c1_hide_path(first_punishment+1:end) = a1.*y1_path(first_punishment+1:end);
c2_hide_path(first_punishment+1:end) = a2.*y2_path(first_punishment+1:end);
Q_hide_path(first_punishment+1:end) = (1-a1).*y1_path(first_punishment+1:end) + y2_path(first_punishment+1:end);
u1_hide_path(first_punishment+1:end) = aut1_path(first_punishment+1:end);
u2_hide_path(first_punishment+1:end) = aut2_path(first_punishment+1:end);

% calculate marriage surplus streams
surplus_hide_path = u1_hide_path + u2_hide_path - aut1_path - aut2_path;

% expected utility at time 0
eu1_hide = mean(u1_hide_path)./(1-beta);
eu2_hide = mean(u2_hide_path)./(1-beta);
eh_hide = (alpha.*mean(u1_hide_path) + (1-alpha).*mean(u2_hide_path))./(1-beta);
eq_hide = (0.5.*mean(u1_hide_path) + 0.5.*mean(u2_hide_path))./(1-beta);
surplus_hide = mean(surplus_hide_path)./(1-beta);

%% IC simulation
% matrices
c1_ic_path = zeros(1,T);
c2_ic_path = zeros(1,T);
Q_ic_path = zeros(1,T);
w_ic_path = zeros(1,T+1);
w_index = zeros(1,T+1);
s_path = zeros(1,T);

% kronecker products
y1_grid = linspace(y1min,y1max,n);
yy1 = kron(y1_grid, ones(1, ns*nfeas*n));
y2_grid = linspace(y2min,y2max,n);
yy2 = kron(ones(1,n), kron(y2_grid, ones(1, ns*nfeas)));
s_grid = linspace(0.1,0.9,ns);
ss = kron(ones(1,n*n), kron(s_grid, ones(1, nfeas)));
ww = kron(ones(1, n*n*ns), w_feasible);
cc1 = a1.*ss.*(yy1+yy2);
cc2 = a2.*(1-ss).*(yy1+yy2);
QQ = (yy1+yy2-cc1-cc2)./price;

% beginning w
w_ic_path(1) = w0;
[~, w_index(1)] = min(abs(w0-w_feasible));

% in each period
for t = 1:T
    % income realization
    y1ind = (yy1 == y1_path(t));
    y2ind = (yy2 == y2_path(t));
    yind = y1ind & y2ind;
    w_realized = ww(yind);
    y1_realized = yy1(yind);
    y2_realized = yy2(yind);
    c1_realized = cc1(yind);
    c2_realized = cc2(yind);
    Q_realized = QQ(yind);
    s_realized = ss(yind);
    % probability vector is determined by last period's future promised
    % utility and this period's income realization
    pi = X_all(yind,w_index(t));
    picum = cumsum(pi).*(n.^2);
    % realization of income, transfer and continuation value
    randnum = rand(1); % random number
    crit1 = (picum > randnum); % random number is less than cumulative distribution
    crit2 = (pi~=0); % points that have mass
    % values that fit criterion
    w_next = w_realized(crit1 & crit2);
    y1_all = y1_realized(crit1 & crit2);
    y2_all = y2_realized(crit1 & crit2);
    c1_all = c1_realized(crit1 & crit2);
    c2_all = c2_realized(crit1 & crit2);
    Q_all = Q_realized(crit1 & crit2);
    s_all = s_realized(crit1 & crit2);
    % update values
    w_ic_path(t+1) = w_next(1);
    c1_ic_path(t) = c1_all(1);
    c2_ic_path(t) = c2_all(1);
    Q_ic_path(t) = Q_all(1);
    s_path(t) = s_all(1);
    % index
    [~, w_index(t+1)] = min(abs(w_feasible-w_ic_path(t+1)));
end

% utility path
u1_ic_path = u1(c1_ic_path,Q_ic_path);
u2_ic_path = u2(c2_ic_path,Q_ic_path);
surplus_ic_path = u1_ic_path + u2_ic_path - aut1_path - aut2_path;

%% Create Graph
figure;

subplot(4,2,1);
plot(1:T,y1_path,1:T,y2_path);
title('Income')

subplot(4,2,2);
plot(1:T, y1_path-z1_path, 1:T, y2_path-z2_path);
title('Hiding')

subplot(4,2,3);
plot(1:T,c1_aut_path,1:T,c1_fb_path,1:T,c1_hide_path,1:T,c1_ic_path);
title('C1')

subplot(4,2,4);
plot(1:T,c2_aut_path,1:T,c2_fb_path,1:T,c2_hide_path,1:T,c2_ic_path);
title('C2')

subplot(4,2,5);
plot(1:T,Q_aut_path,1:T,Q_fb_path,1:T,Q_hide_path,1:T,Q_ic_path);
title('Q')

subplot(4,2,6);
plot(1:T,zeros(1,T),1:T,surplus_fb_path,1:T,surplus_hide_path,1:T,surplus_ic_path);
title('Marital Surplus')
legend('Autarky','First Best', 'Hiding', 'IC')

subplot(4,2,7);
plot(1:T,aut1_path,1:T,u1_fb_path,1:T,u1_hide_path,1:T,u1_ic_path);
title('Utility - Agent 1')

subplot(4,2,8);
plot(1:T,aut2_path,1:T,u2_fb_path,1:T,u2_hide_path,1:T,u2_ic_path);
title('Utility - Agent 2')

%{
%% Graph 2: normalize everything to be a share of first-best
figure;
subplot(1,2,1);
plot(1:T,y1_path,1:T,y2_path);
title('Income')

subplot(1,2,2);
plot(1:T, y1_path-z1_path, 1:T, y2_path-z2_path);
title('Hiding')

figure;

c1_aut_path_normalized = c1_aut_path./c1_fb_path;
c1_hide_path_normalized = c1_hide_path./c1_fb_path;
c1_ic_path_normalized = c1_ic_path./c1_fb_path;
subplot(2,3,1);
plot(1:T,c1_aut_path_normalized,1:T,c1_hide_path_normalized,1:T,c1_ic_path_normalized);
title('C1')

c2_aut_path_normalized = c2_aut_path./c2_fb_path;
c2_hide_path_normalized = c2_hide_path./c2_fb_path;
c2_ic_path_normalized = c2_ic_path./c2_fb_path;
subplot(2,3,2);
plot(1:T,c2_aut_path_normalized,1:T,c2_hide_path_normalized,1:T,c2_ic_path_normalized);
title('C2')

Q_aut_path_normalized = Q_aut_path./Q_fb_path;
Q_hide_path_normalized = Q_hide_path./Q_fb_path;
Q_ic_path_normalized = Q_ic_path./Q_fb_path;
subplot(2,3,3);
plot(1:T,Q_aut_path_normalized,1:T,Q_hide_path_normalized,1:T,Q_ic_path_normalized);
legend('Autarky','Hiding', 'IC')
title('Q')

aut1_path_normalized = aut1_path./u1_fb_path;
u1_hide_path_normalized = u1_hide_path./u1_fb_path;
u1_ic_path_normalized = u1_ic_path./u1_fb_path;
subplot(2,3,4);
plot(1:T,aut1_path_normalized,1:T,u1_hide_path_normalized,1:T,u1_ic_path_normalized);
title('Utility - Agent 1')

aut2_path_normalized = aut2_path./u2_fb_path;
u2_hide_path_normalized = u2_hide_path./u2_fb_path;
u2_ic_path_normalized = u2_ic_path./u2_fb_path;
subplot(2,3,5);
plot(1:T,aut2_path_normalized,1:T,u2_hide_path_normalized,1:T,u2_ic_path_normalized);
title('Utility - Agent 2')

surplus_hide_path_normalized = surplus_hide_path./surplus_fb_path;
surplus_ic_path_normalized = surplus_ic_path./surplus_fb_path;
subplot(2,3,6);
plot(1:T,zeros(1,T),1:T,surplus_hide_path_normalized,1:T,surplus_ic_path_normalized);
title('Marital Surplus')
%}

%% Utility as a function of cost of hiding

% Hiding
z1max = zeros(n_hiding,n_hiding,10);
z2max = zeros(n_hiding,n_hiding,10);
for i = 1:10
    [z1max(:,:,i), z2max(:,:,i)] = infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,deltavar(i),deltavar(i),price,n_hiding);
end

eu1_hide = zeros(1,10);
eu2_hide = zeros(1,10);
surplus_hide = zeros(1,10);
eh_hide = zeros(1,10);
eq_hide = zeros(1,10);

for i = 1:10
% simulate reported income stream
z1_path = zeros(1,T);
z2_path = zeros(1,T);
for t = 1:T
    z1_path(t) = z1max(ind_path1_hiding(t),ind_path2_hiding(t),i);
    z2_path(t) = z2max(ind_path1_hiding(t),ind_path2_hiding(t),i);
end

% simulate consumption and utility stream
c1_hide_path = alpha.*a1.*(z1_path+z2_path) + delta1.*(y1_path-z1_path);
c2_hide_path = (1-alpha).*a2.*(z1_path+z2_path) + delta2.*(y2_path-z2_path);
Q_hide_path = gamma.*(z1_path+z2_path);
u1_hide_path = u1(c1_hide_path,Q_hide_path);
u2_hide_path = u2(c2_hide_path,Q_hide_path);

% simulate probabilities of detection
prob1_path = ((y1_path-z1_path)./y1_path).^2;
prob2_path = ((y2_path-z2_path)./y2_path).^2;

% draw random numbers to see whether punishment state realizes
rand1_path = rand(1,T);
punishment1_path = (rand1_path<prob1_path);
rand2_path = rand(1,T);
punishment2_path = (rand2_path<prob2_path);
% first punishment state
first_punishment1 = find(punishment1_path,1);
first_punishment2 = find(punishment2_path,1);
first_punishment = min(first_punishment1,first_punishment2);

% all time periods after the first punishment state is autarky
c1_hide_path(first_punishment+1:end) = a1.*y1_path(first_punishment+1:end);
c2_hide_path(first_punishment+1:end) = a2.*y2_path(first_punishment+1:end);
Q_hide_path(first_punishment+1:end) = (1-a1).*y1_path(first_punishment+1:end) + y2_path(first_punishment+1:end);
u1_hide_path(first_punishment+1:end) = aut1_path(first_punishment+1:end);
u2_hide_path(first_punishment+1:end) = aut2_path(first_punishment+1:end);

% calculate marriage surplus streams
surplus_hide_path = u1_hide_path + u2_hide_path - aut1_path - aut2_path;

% expected utility at time 0
eu1_hide(i) = mean(u1_hide_path)./(1-beta);
eu2_hide(i) = mean(u2_hide_path)./(1-beta);
eh_hide(i) = (alpha.*mean(u1_hide_path) + (1-alpha).*mean(u2_hide_path))./(1-beta);
eq_hide(i) = (0.5.*mean(u1_hide_path) + 0.5.*mean(u2_hide_path))./(1-beta);
surplus_hide(i) = mean(surplus_hide_path)./(1-beta);
end

%%
figure;
subplot(2,2,1)
plot(deltavar, eu1_hide, deltavar, repmat(P(9),1,10), deltavar, repmat(eu1_fb,1,10))
legend('Hiding','IC','First Best','Location','Southwest')
title('Agent 1 Expected Utility')
subplot(2,2,2)
plot(deltavar, eu2_hide, deltavar, repmat(w0,1,10), deltavar, repmat(eu2_fb,1,10))
title('Agent 2 Expected Utility')
subplot(2,2,3)
eh_ic = alpha.*P(9) + (1-alpha).*w0;
plot(deltavar, eh_hide, deltavar, repmat(eh_ic,1,10), deltavar, repmat(eh_fb,1,10))
title('Household Expected Utility')
eq_ic = 0.5.*P(9) + 0.5.*w0;
subplot(2,2,4)
plot(deltavar, eq_hide, deltavar, repmat(eq_ic,1,10), deltavar, repmat(eq_fb,1,10))
title('Equally-weighted Expected Utility')