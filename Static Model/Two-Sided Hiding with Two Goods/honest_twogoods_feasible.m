function [exphhutility, alloc_diff, res1, res2] = honest_twogoods_feasible(z, res1, res2, a1, a2, alpha, rho, price, delta1o, delta1u, delta2o, delta2u, inc, prob)

res1(2,1) = z(1);
res1(3,1) = z(2);
res2(1,2) = z(3);
res2(1,3) = z(4);

%% Calculate minimum allocation for agent 1
% State 2 for agent 1, state 2 for agent 2
maxu = @(x3) besthiding(x3, a1, res1(2,1), rho, price, delta1o, delta1u);
init = (1-a1).*delta1u./price;
options = optimoptions('fmincon','Display','off');
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta1u.*price),[],options);
% equivalent allocation in honest equilibrium
res1(2,2) = exp(-u + (1-a1).*log(price) - a1.*log(a1) - (1-a1).*log(1-a1));

% State 2 for agent 1, state 3 for agent 2
maxu = @(x3) besthiding(x3, a1, res1(3,1), rho, price, delta1o, delta1u);
init = (1-a1).*delta1u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta1u.*price),[],options);
% equivalent allocation in honest equilibrium
res1(3,2) = exp(-u + (1-a1).*log(price) - a1.*log(a1) - (1-a1).*log(1-a1));

% State 3 for agent 1, state 2 for agent 2
maxu = @(x3) besthiding(x3, a1, res1(2,2), rho, price, delta1o, delta1u);
init = (1-a1).*delta1u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta1u.*price),[],options);
% equivalent allocation in honest equilibrium
res1(2,3) = exp(-u + (1-a1).*log(price) - a1.*log(a1) - (1-a1).*log(1-a1));

% State 3 for agent 1, state 3 for agent 2
maxu = @(x3) besthiding(x3, a1, res1(3,2), rho, price, delta1o, delta1u);
init = (1-a1).*delta1u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta1u.*price),[],options);
% equivalent allocation in honest equilibrium
res1(3,3) = exp(-u + (1-a1).*log(price) - a1.*log(a1) - (1-a1).*log(1-a1));

%% Calculate minimum allocation for agent 2
% State 2 for agent 1, state 2 for agent 2
maxu = @(x3) besthiding(x3, a2, res2(1,2), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta2u.*price),[],options);
% equivalent allocation in honest equilibrium
res2(2,2) = exp(-u + (1-a2).*log(price) - a2.*log(a2) - (1-a2).*log(1-a2));

% State 3 for agent 1, state 2 for agent 2
maxu = @(x3) besthiding(x3, a2, res2(1,3), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta2u.*price),[],options);
% equivalent allocation in honest equilibrium
res2(2,3) = exp(-u + (1-a2).*log(price) - a2.*log(a2) - (1-a2).*log(1-a2));

% State 2 for agent 1, state 3 for agent 2
maxu = @(x3) besthiding(x3, a2, res2(2,2), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta2u.*price),[],options);
% equivalent allocation in honest equilibrium
res2(3,2) = exp(-u + (1-a2).*log(price) - a2.*log(a2) - (1-a2).*log(1-a2));

% State 3 for agent 1, state 3 for agent 2
maxu = @(x3) besthiding(x3, a2, res2(2,3), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
[~,u] = fmincon(maxu,init,[],[],[],[],0,1./(delta2u.*price),[],options);
% equivalent allocation in honest equilibrium
res2(3,3) = exp(-u + (1-a2).*log(price) - a2.*log(a2) - (1-a2).*log(1-a2));

% Calculate feasibility
alloc_diff = round(inc - res1 - res2, 4);
% over-allocated
alloc_diff_neg = max(max(alloc_diff<0));
% Calculate utility
utility1 = log(res1) - (1-a1).*log(price) + a1.*log(a1) + (1-a1).*log(1-a1);
utility2 = log(res2) - (1-a2).*log(price) + a1.*log(a2) + (1-a2).*log(1-a2);
% household utility
hhutility = alpha.*utility1 + (1-alpha).*utility2;
% (negative) expected utility
exphhutility = -sum(sum(prob.*hhutility)) + alloc_diff_neg.*999999;

end