%% Is hiding equilibrium incentive compatible?
% To answer Alessandra's question of why incentive compatible equilibrium
% does not always dominate hiding / why revelation principle wouldn't
% apply, I want to solve for a hiding equilibrium, modify it to Pareto
% efficient, and check whether that new allocation is incentive compatible.

function [u1_pol, u2_pol, u1_ic, u2_ic] = is_hiding_ic(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,price,n,delta1,delta2)

%clear all; close all; clc;

%% Parameters
%{
% set seed
rng(2);

% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;

% price of public good
price = 1;

% Two income processes: uniform
y1min = 1; y1max = 5;
y2min = 1; y2max = 5;

% Pareto weight
alpha = 0.3;

% Time discounting
beta = 0.94;

% Cost of hiding
delta1 = 0.9; delta2 = 0.9;

% Discretize income space
n = 100;
%}

% probability of punishment
pi = @(y, haty) ((y-haty)./y).^2;

%% Set up

% utility
gamma = (1-a1).*alpha + (1-a2).*(1-alpha);
% each agent's utility depend on their actual income and both agents'
% reported income
u1 = @(y,z1,z2) a1.*log(alpha.*a1.*(z1+z2) + delta1.*(y-z1)) + (1-a1).*log(gamma.*(z1+z2)./price);
u2 = @(y,z1,z2) a2.*log((1-alpha).*a2.*(z1+z2) + delta2.*(y-z2)) + (1-a2).*log(gamma.*(z1+z2)./price);

% discretize income space
income1 = linspace(y1min,y1max,n);
income2 = linspace(y2min,y2max,n);

% Autarky value
% autarky utility
uaut1 = @(y) a1.*log(a1.*y)+(1-a1).*log((1-a1).*y./price);
uaut2 = @(y) a2.*log(a2.*y)+(1-a2).*log((1-a2).*y./price);
% calculate expected utility under forever autarky
vaut1 = mean(uaut1(income1))./(1-beta);
vaut2 = mean(uaut2(income2))./(1-beta);

% error tolerance
error = 1e-4;

%% Agent 1
% set up matrices for agent 1
maxv1 = zeros(n);
maxzind1 = zeros(n);
% guess of initial continuation value
diff = 1;
V_exp1 = vaut1+1;

% Agent 1
tic
while diff>error
    % for each self income realization
    for i = 1:n
        % and each reported income of the other agent
        for j = 1:n
        % possible reported income is not greater than actual income
        z_test = income1(1:i)';
        % current period utility
        u1_test = u1(income1(i),z_test,income2(j));
        % continuation value
        V_test = u1_test + beta.*vaut1.*pi(income1(i),z_test) + beta.*(1-pi(income1(i),z_test)).*V_exp1;
        % choose reported income to maxize continuation value
        [maxv1(i,j), maxzind1(i,j)] = max(V_test);
        end
    end
    diff = abs(V_exp1 - mean(mean(maxv1)));
    V_exp1 = mean(mean(maxv1));
end
toc

%% Agent 2
% set up matrices for agent 2
maxv2 = zeros(n);
maxzind2 = zeros(n);

% initial guess of continuation value
diff = 1;
V_exp2 = vaut2+1;

tic
% Agent 2
while diff>error
    % for each self income realization
    for i = 1:n
    %for i = 1:n
        % and each reported income of the other agent
        for j = 1:n
        % possible reported income is not greater than actual income
        z_test = income2(1:i)';
        % current period utility
        u2_test = u2(income2(i),income1(j),z_test);
        % continuation value
        V_test = u2_test + beta.*vaut2.*pi(income2(i),z_test) + beta.*(1-pi(income2(i),z_test)).*V_exp2;
        % choose reported income to maximize continuation value
        [maxv2(i,j), maxzind2(i,j)] = max(V_test);
        end
    end
    diff = abs(V_exp2 - mean(mean(maxv2)));
    V_exp2 = mean(mean(maxv2));
end
toc

%% Find best response
z1 = zeros(n);
z2 = zeros(n);
tic
% for each agent 1's income realization
for i = 1:n
    % and each agent 2's income realization
    for j = 1:n
        % guess a response function
        z1(i,j) = i;
        z2(i,j) = j;
        max_diff = 100;
        while max_diff>1
            % calculate difference between current guess and best response
            z1_diff = abs(z1(i,j)-maxzind1(i,z2(i,j)));
            z2_diff = abs(z2(i,j)-maxzind2(j,z1(i,j)));
            max_diff = max(z1_diff,z2_diff);
            % update
            z1(i,j) = maxzind1(i,z2(i,j));
            z2(i,j) = maxzind2(j,z1(i,j));
        end
    end
end
toc

%% Policy functions
% Reported income as a function of realized income
z1max = income1(z1);
z2max = income2(z2);

% consumption
income1_rep = repmat(income1',1,n);
income2_rep = repmat(income2,n,1);
c1 = a1.*alpha.*(z1max + z2max) + delta1.*(income1_rep - z1max);
c2 = a2.*(1-alpha).*(z1max + z2max) + delta2.*(income2_rep - z2max);
Q = gamma.*(z1max + z2max);

% sharing rule
c1_sharing = c1./(c1+c2);
mean_c1_sharing = mean(mean(c1_sharing));
c2_sharing = c2./(c1+c2);
mean_c2_sharing = mean(mean(c2_sharing));

% utility
u1_pol = u1(income1_rep,z1max,z2max);
u2_pol = u2(income2_rep,z1max,z2max);
u1_pol_mean = mean(reshape(u1_pol,1,[])); 
u2_pol_mean = mean(reshape(u2_pol,1,[])); 

%% Modify the allocation to be Pareto efficient
% hiding
hiding1 = (income1_rep - z1max>0);
hiding2 = (income2_rep - z2max>0);
% add loss consumption to the others' private consumption
c1_ic = c1;
c1_ic(hiding2) = c1(hiding2) + (1-delta2).*(income2_rep(hiding2) - z2max(hiding2));
c2_ic = c2;
c2_ic(hiding1) = c2(hiding1) + (1-delta1).*(income1_rep(hiding1) - z1max(hiding1));

% sharing rule
c1_ic_sharing = c1_ic./(c1_ic+c2_ic);
mean_c1_ic_sharing = mean(mean(c1_ic_sharing));
c2_ic_sharing = c2_ic./(c1_ic+c2_ic);
mean_c2_ic_sharing = mean(mean(c2_ic_sharing));

% utility
u1_ic = a1.*log(c1_ic) + (1-a1).*log(Q);
u2_ic = a2.*log(c2_ic) + (1-a2).*log(Q);
u1_ic_mean = mean(reshape(u1_ic,1,[]));
u2_ic_mean = mean(reshape(u2_ic,1,[]));

%% Is the new allocation incentive compatible
% agent 1
counter = 1;
% for each income realization of agent 1
for i = 2:n
    % for each income realization that is lower
    for j = 1:i-1
        % for each income realization of agent 2
        for k = 1:n
            % utility of being truthful
            truthful_1 = a1.*log(c1_ic(i,k)) + (1-a1).*log(Q(i,k));
            % utility of hiding
            lying_1 = a1.*log(c1_ic(j,k)+delta1*(c1_ic(i,k)-c1_ic(j,k))) + (1-a1).*log(Q(j,k));
            % is being truthful better?
            ic1(counter) = (truthful_1<=lying_1); 
            % add counter
            counter = counter+1;
        end
    end
end
sum_ic1 = sum(ic1);

% agent 2
counter = 1;
% for each income realization of agent 2
for i = 2:n
    % for each income realization that is lower
    for j = 1:i-1
        % for each income realization of agent 1
        for k = 1:n
            % utility of being truthful
            truthful_2 = a2.*log(c2_ic(k,i)) + (1-a2).*log(Q(k,i));
            % utility of hiding
            lying_2 = a2.*log(c2_ic(k,j)+delta2*(c2_ic(k,i)-c2_ic(k,i))) + (1-a2).*log(Q(k,j));
            % is being truthful better?
            ic2(counter) = (truthful_2<=lying_2); 
            % add counter
            counter = counter+1;
        end
    end
end
sum_ic2 = sum(ic2);
