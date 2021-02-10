function [exphhutility, res1, res2, alloc_diff] = honest_twogoods(x, a1, a2, rho, price, delta1o, delta1u, delta2o, delta2u, inc, alpha, prob)

%% Set up - uncomment to run as script
%{
% Parameters
% cobb-douglas utility function parameter
% utility = a*log(observable) + (1-a)*log(unobservable)
a1 = 0.5;
a2 = 0.5;
% price for the unobservable good (price of the observable good is
% normalized to 1)
price = 1;
% 4 hiding parameters: for two goods and two agents
% 1 unit of hidden income can be used as delta expenditure toward a good
% observable delta < unobservable delta
delta1o = 0.8;
delta2o = 0.8;
delta1u = 0.1;
delta2u = 0.1;
% alpha [0,1] is the Pareto weight of agent 1.
alpha = 0.3;
% Agent 2's income is normalized to 0,1,2 in the three states. Agent 1's
% income are y0, y0+rho, y0+2rho.
y0 = 1;
rho = 1;
% p is the probability of states for agent 1, q is the probability of
% states for agent 2;
p = [0.33 0.33 0.33];
q = [0.33;0.33;0.33];

% Calculate income in all states and probability of states
y1 = [y0 y0+rho y0+2*rho];
y2 = [1;2;3];
inc = y1+y2;
x = [alpha.*inc(1,1);(1-alpha).*inc(1,1)];
prob = p.*q;
%}

%% Set up matrices
% Create two 3-by-3 matrices to store resource allocation
res1 = zeros(3);
res2 = zeros(3);

% Initialization: resource allocation in the (1,1) state
res1(1,1) = x(1);
res2(1,1) = x(2);

%% Holding constant state 1 for agent 2, calculate all allocations for different states of agent 1
% In state 2 for agent 1, find agent 1's max utility through hiding
maxu = @(x2) besthiding(x2, a1, res1(1,1), rho, price, delta1o, delta1u);
init = rho.*(1-a1).*delta1u./price;
options = optimoptions('fmincon','Display','off');
[~,u] = fmincon(maxu,init,[],[],[],[],0,rho./(delta1u.*price),[],options);
% equivalent allocation in honest equilibrium
res1(1,2) = exp(-u + (1-a1).*log(price) - a1.*log(a1) - (1-a1).*log(1-a1));
%res2(1,2) = inc(1,2) - res1(1,2);

% In state 3 for agent 1, find agent 1's max utility through hiding
maxu = @(x3) besthiding(x3, a1, res1(1,2), rho, price, delta1o, delta1u);
init = rho.*(1-a1).*delta1u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,rho./(delta1u.*price),[],options);
% equivalent allocation in honest equilibrium
res1(1,3) = exp(-u + (1-a1).*log(price) - a1.*log(a1) - (1-a1).*log(1-a1));
%res2(1,3) = inc(1,3) - res1(1,3);

%% Holding constant state 1 for agent 1, calculate all states of agent 2
% State 1 for agent 1, state 2 for agent 2
maxu = @(x3) besthiding(x3, a2, res2(1,1), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
[~, u] = fmincon(maxu,init,[],[],[],[],0,1./(delta2u.*price),[],options);
% equivalent allocation in honest equilibrium
res2(2,1) = exp(-u + (1-a2).*log(price) - a2.*log(a2) - (1-a2).*log(1-a2));
%res1(2,1) = inc(2,1) - res2(2,1);

% State 1 for agent 1, state 3 for agent 2
maxu = @(x3) besthiding(x3, a2, res2(2,1), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta2u.*price),[],options);
% equivalent allocation in honest equilibrium
res2(3,1) = exp(-u + (1-a2).*log(price) - a2.*log(a2) - (1-a2).*log(1-a2));
%res1(3,1) = inc(3,1) - res2(3,1);

%% Find feasible allocation
fea = @(z) honest_twogoods_feasible(z, res1, res2, a1, a2, alpha, rho, price, delta1o, delta1u, delta2o, delta2u, inc, prob);
zmax = [inc(2,1) - res2(2,1);inc(3,1) - res2(3,1);inc(1,2) - res1(1,2);inc(1,3) - res1(1,3)];
z0 = zmax./2;
%options = optimoptions('fmincon','Display','iter');
rng default % For reproducibility
opts = optimoptions(@fmincon,'Algorithm','sqp');
problem = createOptimProblem('fmincon','objective',fea,'x0',z0,'lb',[0;0;0;0],'ub',zmax,'options',opts);
ms = MultiStart;
[zfea,~] = run(ms,problem,4);
[exphhutility, alloc_diff, res1, res2] = honest_twogoods_feasible(zfea, res1, res2, a1, a2, alpha, rho, price, delta1o, delta1u, delta2o, delta2u, inc, prob);

end