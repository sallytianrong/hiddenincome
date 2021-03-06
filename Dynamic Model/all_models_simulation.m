% Simulation of first-best, incentive compatible, and hiding equilibrium
% This code uses a common set of parameters, solve for both incentive
% compative and hiding equilibria using functions, and use those outputs
% for a simulation.

clear all; close all; clc;

tic

%% Parameters
% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;

% Two income processes: uniform discrete
y1min = 2; y1max = 3;
y2min = 2; y2max = 6;

% Pareto weight
alpha = 0.3;

% price
price = 1;

% Time discounting
beta = 0.94;

% discretize income space
n = 5;
income1 = linspace(y1min,y1max,n);
income2 = linspace(y2min,y2max,n);

% discretize state space for IC
ns = 20; nw = 20;

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

% probability of punishment
punish = @(y, haty) ((y-haty)./y).^2;

%% Solve incentive compatible
[w_feasible, P, nfeas, X_all, ec1_ic, ec2_ic, eQ_ic, eu1_ic, eu2_ic] = IC_twoside_whiding(a1,a2,y1min,y1max,y2min,y2max,beta,price,n,ns,nw,delta1,delta2,punish);

% save workspace because this takes a long time to run! it will be easier
% if I don't need to run it every time
save ('ic.mat');

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

%% Hiding

% load workspace
load ('ic.mat');

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
[z1max_bl, z2max_bl, c1_hiding, c2_hiding, Q_hiding, u1_hiding, u2_hiding] = infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,delta1,delta2,price,n_hiding,punish);

%% Policy functions
% Private consumption of agent 1
%mpc1_aut = a1;
%mpc1_fb = a1.*alpha;
%mpc1_hide_lowy2 = mean((c1_hiding(2:end,1) - c1_hiding(1:end-1,1))./(income1_hiding(2:end) - income1_hiding(1:end-1))');
%mpc1_hide_highy2 = mean((c1_hiding(2:end,100) - c1_hiding(1:end-1,100))./(income1_hiding(2:end) - income1_hiding(1:end-1))');
%mpc1_ic_lowy2loww = (ec1_ic(2:end,1,1) - ec1_ic(1:end-1,1,1))./(income1(2:end) - income1(1:end-1));

%% Simulation
rng(8);

% simulation time periods
T = 500;

% generate alternate income path
%ind_path1 = repmat([1 3],1,T/2);
%ind_path2 = repmat([3 1],1,T/2);

% Simulate income processes
ind_path1 = randi([1 n],T,1);
ind_path2 = randi([1 n],T,1);
ind_path1_hiding = map(ind_path1);
ind_path2_hiding = map(ind_path2);
y1_path = income1(ind_path1);
y2_path = income2(ind_path2);

%% Autarky
% autarky consumption path
c1_aut_path = a1.*y1_path;
c2_aut_path = a2.*y2_path;
s_aut_path = c1_aut_path./(c1_aut_path+c2_aut_path);
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
s_fb_path = c1_fb_path./(c1_fb_path+c2_fb_path);
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
s_hide_path = c1_hide_path./(c1_hide_path+c2_hide_path);
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

% Maximize social planner's expected utility at period 0
social_planner = alpha.*P + (1-alpha).*w_feasible;
[w0, w_index(1)] = max(social_planner);
w_ic_path(1) = w0;

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
s_ic_path = c1_ic_path./(c1_ic_path+c2_ic_path);
u1_ic_path = u1(c1_ic_path,Q_ic_path);
u2_ic_path = u2(c2_ic_path,Q_ic_path);
surplus_ic_path = u1_ic_path + u2_ic_path - aut1_path - aut2_path;

%% Risk sharing
% variance of income
var_y1 = var(y1_path);
var_y2 = var(y2_path);
% variance of consumption (agent 1)
var_c1_aut = var(c1_aut_path);
var_c1_fb = var(c1_fb_path);
var_c1_hide = var(c1_hide_path);
var_c1_ic = var(c1_ic_path);
% variance of consumption (agent 2)
var_c2_aut = var(c2_aut_path);
var_c2_fb = var(c2_fb_path);
var_c2_hide = var(c2_hide_path);
var_c2_ic = var(c2_ic_path);
% variance of consumption (Q)
var_Q_aut = var(Q_aut_path);
var_Q_fb = var(Q_fb_path);
var_Q_hide = var(Q_hide_path);
var_Q_ic = var(Q_ic_path);
% variance of utility (agent 1)
var_u1_aut = var(aut1_path);
var_u1_fb = var(u1_fb_path);
var_u1_hide = var(u1_hide_path);
var_u1_ic = var(u1_ic_path);
% variance of utility (agent 2)
var_u2_aut = var(aut2_path);
var_u2_fb = var(u2_fb_path);
var_u2_hide = var(u2_hide_path);
var_u2_ic = var(u2_ic_path);

%% Simulated Policy Functions
for i = 1:n
    for j = 1:n
        y1index = (ind_path1 == i);
        y2index = (ind_path2 == j);
        c1_hide_emp(i,j) = mean(c1_hide_path(y1index & y2index));
        c2_hide_emp(i,j) = mean(c2_hide_path(y1index & y2index));
        Q_hide_emp(i,j) = mean(c1_hide_path(y1index & y2index));
    end
end

%% Create Simulation Graph
figure;

subplot(2,4,1);
plot(1:50,y1_path(1:50),1:50,y2_path(1:50));
title('Income')

subplot(2,4,5);
plot(1:50, y1_path(1:50)-z1_path(1:50), 1:50, y2_path(1:50)-z2_path(1:50));
title('Hiding')

subplot(2,4,3);
plot(1:50,c1_aut_path(1:50),1:50,c1_fb_path(1:50),1:50,c1_hide_path(1:50),1:50,c1_ic_path(1:50));
title('C1')

subplot(2,4,7);
plot(1:50,c2_aut_path(1:50),1:50,c2_fb_path(1:50),1:50,c2_hide_path(1:50),1:50,c2_ic_path(1:50));
title('C2')

subplot(2,4,4);
plot(1:50,Q_aut_path(1:50),1:50,Q_fb_path(1:50),1:50,Q_hide_path(1:50),1:50,Q_ic_path(1:50));
title('Q')
l=legend('Autarky / Agent 1','First Best / Agent 2','Hiding','IC')
set(l, 'Position', [0.62 0.2 0.4 0.2])

subplot(2,4,2);
plot(1:50,aut1_path(1:50),1:50,u1_fb_path(1:50),1:50,u1_hide_path(1:50),1:50,u1_ic_path(1:50));
title('Utility - Agent 1')

subplot(2,4,6);
plot(1:50,aut2_path(1:50),1:50,u2_fb_path(1:50),1:50,u2_hide_path(1:50),1:50,u2_ic_path(1:50));
title('Utility - Agent 2')

%% Graph 2: normalize everything to be a share of first-best
figure;

c1_aut_path_normalized = c1_aut_path./c1_fb_path;
c1_hide_path_normalized = c1_hide_path./c1_fb_path;
c1_ic_path_normalized = c1_ic_path./c1_fb_path;
subplot(2,3,1);
plot(1:100,c1_aut_path_normalized(1:100),1:100,c1_hide_path_normalized(1:100),1:100,c1_ic_path_normalized(1:100));
title('C1')

c2_aut_path_normalized = c2_aut_path./c2_fb_path;
c2_hide_path_normalized = c2_hide_path./c2_fb_path;
c2_ic_path_normalized = c2_ic_path./c2_fb_path;
subplot(2,3,2);
plot(1:100,c2_aut_path_normalized(1:100),1:100,c2_hide_path_normalized(1:100),1:100,c2_ic_path_normalized(1:100));
title('C2')

Q_aut_path_normalized = Q_aut_path./Q_fb_path;
Q_hide_path_normalized = Q_hide_path./Q_fb_path;
Q_ic_path_normalized = Q_ic_path./Q_fb_path;
subplot(2,3,3);
plot(1:100,Q_aut_path_normalized(1:100),1:100,Q_hide_path_normalized(1:100),1:100,Q_ic_path_normalized(1:100));
legend('Autarky','Hiding', 'IC')
title('Q')

aut1_path_normalized = aut1_path./u1_fb_path;
u1_hide_path_normalized = u1_hide_path./u1_fb_path;
u1_ic_path_normalized = u1_ic_path./u1_fb_path;
subplot(2,3,4);
plot(1:100,aut1_path_normalized(1:100),1:100,u1_hide_path_normalized(1:100),1:100,u1_ic_path_normalized(1:100));
title('Utility - Agent 1')

aut2_path_normalized = aut2_path./u2_fb_path;
u2_hide_path_normalized = u2_hide_path./u2_fb_path;
u2_ic_path_normalized = u2_ic_path./u2_fb_path;
subplot(2,3,5);
plot(1:100,aut2_path_normalized(1:100),1:100,u2_hide_path_normalized(1:100),1:100,u2_ic_path_normalized(1:100));
title('Utility - Agent 2')

surplus_hide_path_normalized = surplus_hide_path./surplus_fb_path;
surplus_ic_path_normalized = surplus_ic_path./surplus_fb_path;
subplot(2,3,6);
plot(1:100,zeros(1,100),1:100,surplus_hide_path_normalized(1:100),1:100,surplus_ic_path_normalized(1:100));
title('Marital Surplus')

%% Presentation graph
figure;

c1_aut_path_normalized = c1_aut_path-c1_fb_path;
c1_hide_path_normalized = c1_hide_path-c1_fb_path;
c1_ic_path_normalized = c1_ic_path-c1_fb_path;
subplot(1,3,1);
plot(1:50,zeros(1,50),1:50,c1_ic_path_normalized(1:50),1:50,c1_aut_path_normalized(1:50),1:50,c1_hide_path_normalized(1:50));
title('C1')

c2_aut_path_normalized = c2_aut_path-c2_fb_path;
c2_hide_path_normalized = c2_hide_path-c2_fb_path;
c2_ic_path_normalized = c2_ic_path-c2_fb_path;
subplot(1,3,2);
plot(1:50,zeros(1,50),1:50,c2_ic_path_normalized(1:50),1:50,c2_aut_path_normalized(1:50),1:50,c2_hide_path_normalized(1:50));
title('C2')

Q_aut_path_normalized = Q_aut_path-Q_fb_path;
Q_hide_path_normalized = Q_hide_path-Q_fb_path;
Q_ic_path_normalized = Q_ic_path-Q_fb_path;
subplot(1,3,3);
plot(1:50,zeros(1,50),1:50,Q_ic_path_normalized(1:50),1:50,Q_aut_path_normalized(1:50),1:50,Q_hide_path_normalized(1:50));
legend('First Best','IC','Autarky', 'Hiding')
title('Q')

%%
figure;

aut1_path_normalized = aut1_path-u1_fb_path;
u1_hide_path_normalized = u1_hide_path-u1_fb_path;
u1_ic_path_normalized = u1_ic_path-u1_fb_path;
subplot(1,4,1);
plot(1:50,zeros(1,50),1:50,u1_ic_path_normalized(1:50),1:50,aut1_path_normalized(1:50),1:50,u1_hide_path_normalized(1:50));
title('Utility - Agent 1')
ylim([-0.4 0.3]);

aut2_path_normalized = aut2_path-u2_fb_path;
u2_hide_path_normalized = u2_hide_path-u2_fb_path;
u2_ic_path_normalized = u2_ic_path-u2_fb_path;
subplot(1,4,2);
plot(1:50,zeros(1,50),1:50,u2_ic_path_normalized(1:50),1:50,aut2_path_normalized(1:50),1:50,u2_hide_path_normalized(1:50));
title('Utility - Agent 2')
ylim([-0.4 0.3]);

authh_path_normalized = (alpha.*aut1_path + (1-alpha).*aut2_path)-(alpha.*u1_fb_path + (1-alpha).*u2_fb_path);
hidehh_path_normalized = (alpha.*u1_hide_path + (1-alpha).*u2_hide_path)-(alpha.*u1_fb_path + (1-alpha).*u2_fb_path);
ichh_path_normalized = (alpha.*u1_ic_path + (1-alpha).*u2_ic_path)-(alpha.*u1_fb_path + (1-alpha).*u2_fb_path);

subplot(1,4,3);
plot(1:50,zeros(1,50),1:50,ichh_path_normalized(1:50),1:50,authh_path_normalized(1:50),1:50,hidehh_path_normalized(1:50));
title('Household Utility (Pareto Weighted)')
ylim([-0.4 0.1]);

auteq_path_normalized = (0.5.*aut1_path + (1-0.5).*aut2_path)-(0.5.*u1_fb_path + (1-0.5).*u2_fb_path);
hideeq_path_normalized = (0.5.*u1_hide_path + (1-0.5).*u2_hide_path)-(0.5.*u1_fb_path + (1-0.5).*u2_fb_path);
iceq_path_normalized = (0.5.*u1_ic_path + (1-0.5).*u2_ic_path)-(0.5.*u1_fb_path + (1-0.5).*u2_fb_path);

subplot(1,4,4);
plot(1:50,zeros(1,50),1:50,iceq_path_normalized(1:50),1:50,auteq_path_normalized(1:50),1:50,hideeq_path_normalized(1:50));
title('Household Utility (Equally Weighted)')
legend('First Best','IC','Autarky', 'Hiding', 'Location', 'Southeast')
ylim([-0.4 0.1]);

%% Mean Graph
figure;

subplot(2,3,1);
cat = categorical({'Agent 1','Agent 2'});
cat = reordercats(cat,{'Agent 1','Agent 2'});
b = bar(cat,[mean(y1_path) mean(y2_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
title('Mean(Income)')

subplot(2,3,2);
cat = categorical({'Autarky','First Best','Hiding','IC'});
cat = reordercats(cat,{'Autarky','First Best','Hiding','IC'});
b = bar(cat,[mean(aut1_path) mean(u1_fb_path) mean(u1_hide_path) mean(u1_ic_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
b.CData(4,:) = [0.67 0.93 0.85];
title('Mean(U1)')

subplot(2,3,3);
b = bar(cat,[mean(aut2_path) mean(u1_fb_path) mean(u2_hide_path) mean(u2_ic_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
b.CData(4,:) = [0.67 0.93 0.85];
title('Mean(U2)')

subplot(2,3,4);
b = bar(cat,[mean(c1_aut_path) mean(c1_fb_path) mean(c1_hide_path) mean(c1_ic_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
b.CData(4,:) = [0.67 0.93 0.85];
title('Mean(C1)')

subplot(2,3,5);
b = bar(cat,[mean(c2_aut_path) mean(c2_fb_path) mean(c2_hide_path) mean(c2_ic_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
b.CData(4,:) = [0.67 0.93 0.85];
title('Mean(C2)')

subplot(2,3,6);
b = bar(cat,[mean(Q_aut_path) mean(Q_fb_path) mean(Q_hide_path) mean(Q_ic_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
b.CData(4,:) = [0.67 0.93 0.85];
title('Mean(Q)')

%% Presentation graph
figure;

subplot(1,3,1);
cat = categorical({'Agent 1','Agent 2'});
cat = reordercats(cat,{'Agent 1','Agent 2'});
b = bar(cat,[mean(y1_path) mean(y2_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
title('Mean(Income)')

subplot(1,3,2);
cat = categorical({'Autarky','First Best','Hiding','IC'});
cat = reordercats(cat,{'Autarky','First Best','Hiding','IC'});
b = bar(cat,[mean(aut1_path) mean(u1_fb_path) mean(u1_hide_path) mean(u1_ic_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
b.CData(4,:) = [0.67 0.93 0.85];
title('Mean(U1)')
ylim([-0.1 1])

subplot(1,3,3);
b = bar(cat,[mean(aut2_path) mean(u2_fb_path) mean(u2_hide_path) mean(u2_ic_path)]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
b.CData(4,:) = [0.67 0.93 0.85];
title('Mean(U2)')
ylim([-0.1 1])

%% Risk sharing graph
figure;

subplot(2,3,1);
cat = categorical({'Agent 1','Agent 2'});
cat = reordercats(cat,{'Agent 1','Agent 2'});
b = bar(cat,[var_y1 var_y2]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
title('Var(Income)')

subplot(2,3,2);
cat = categorical({'First Best','Hiding','IC'});
cat = reordercats(cat,{'First Best','Hiding','IC'});
b = bar(cat,[var_u1_fb var_u1_hide var_u1_ic]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
title('Var(U1)')

subplot(2,3,3);
b = bar(cat,[var_u2_fb var_u2_hide var_u2_ic]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
title('Var(U2)')

subplot(2,3,4);
b = bar(cat,[var_c1_fb var_c1_hide var_c1_ic]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
title('Var(C1)')

subplot(2,3,5);
b = bar(cat,[var_c2_fb var_c2_hide var_c2_ic]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
title('Var(C2)')

subplot(2,3,6);
b = bar(cat,[var_Q_fb var_Q_hide var_Q_ic]);
b.FaceColor = 'flat';
b.CData(1,:) = [0.28 0.27 0.43];
b.CData(2,:) = [0.24 0.52 0.66];
b.CData(3,:) = [0.27 0.8 0.81];
title('Var(Q)')

%% Inequality

% sharing rule as a function of Pareto weight
alpha_var = linspace(0.1,0.9,10);

% first best: sharing rule = Pareto weight

% hiding
seed = 8;
for i = 1:10
    [s_hide(i), s_ic(i)] = simulation_func(seed,T,n,map,income1,income2,a1,a2,uaut1,uaut2,u1,u2,beta,alpha_var(i),delta1,delta2,price,z1max_bl,z2max_bl,P,w_feasible,X_all,ns,nfeas);
end

figure;
plot(alpha_var,alpha_var,alpha_var,s_hide,alpha_var,s_ic);
xlabel('Pareto Weight')
ylabel('Sharing Rule')
title('Mean Sharing Rule')
legend('First Best','Hiding','IC','Location','Southeast')
toc

%{
%% Utility as a function of cost of hiding

% Hiding
z1max = zeros(n_hiding,n_hiding,10);
z2max = zeros(n_hiding,n_hiding,10);
for i = 1:10
    [z1max(:,:,i), z2max(:,:,i)] = infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,deltavar(i),deltavar(i),price,n_hiding,punish);
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
plot(deltavar, eu1_hide, deltavar, repmat(P(2),1,10), deltavar, repmat(eu1_fb,1,10))
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
%}
