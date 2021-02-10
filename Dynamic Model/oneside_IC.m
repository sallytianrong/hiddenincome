% Incentive compatible allocations under one sided hiding
% key elements: 1) consumption allocation 2) future expected utility

%% Parameters
% set seed
rng(4);

% Parameters
% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;

% Pareto weight
alpha = 0.5;

% Time discounting
beta = 0.96;

% Cost of hiding
delta = 0.8;

% agent 2's income
y2 = 3;

%% Functional forms
% Income process is uniform on the interval [ymin,ymax]. agent 2's income
% is fixed.
ymin = 1; ymax = 5;

% to simplify, let's say there are only two states of income, with equal
% probability
n = 10;
income = [ymin ymax];

% grid of promised utility values is bounded below by autarky value and
% above by the agent 1 consume the entire endowment every period
V = linspace(log(ymin)./(1-beta),log(ymax+y2)./(1-beta),n);

% agent 2 chooses how much to transfer to agent 1 (can be negative)
% the most they can transfer is all of agent 2's income. the least they can
% transfer is taking all of agent 1's income.
transfer = linspace(-ymax,y2,n);

%% Find optimal T
% transfer is a function of V(i) and income(j), transfer must increase in
% both
T = zeros(n,2);

%% for each V and T, calculate W schedule
% Continuation utility of agent 1
W = zeros(n,2);
W_index = zeros(n,2);

% matrices to store interim results
W_test = zeros(1,2);
V_test = zeros(1,2);

% for each V gridpoint
for i = 1:n
    % loop through all gridpoints to find starting value of W
    for j = 1:n        
        % starting value of W
        W_test(1) = V(j);

        % the rest of W can be found by downward binding IC constraint
        for t = 2
            W_test(t) = log(income(t) + T(i,t-1))./beta + W_test(t-1) - log(income(t) + T(i,t))./beta;
        end

        % calculate promised utility
        V_test(j) = mean(log(income+T(i,:)) + beta.*W_test);
    end
    
    % find the appropriate starting point of W
    [~, index] = min(V_test - V(i));
    
    % calculate W for each V gridpoint
    W(i,1) = W_test(index);
    W_index(i,1) = index;
    for t = 2
        W(i,t) = log(income(t) + T(i,t-1))./beta + W(i,t-1) - log(income(t) + T(i,t))./beta;
        [~, W_index(i,t)] = min(abs(W(i,t)-V));
    end
end

%% Taking V as given, agent 2 maximizes utility
% value function of agent 2
P = mean(W,2);
diff = 1;

while diff>0.001
    P_update = mean(log(y2 - T) + beta.*P(W_index),2);
    diff = max(abs(P_update-P));
    P = P_update;
end

