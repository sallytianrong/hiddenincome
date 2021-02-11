function [eu, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_bilateral_public_good3(x,a1,a2,price,delta1,delta2,alpha,y,rho,prob)

%% Matrices to store allocation
c_hc = zeros(3);
d_hc = zeros(3);
Q_hc = zeros(3);
u_hc = zeros(3);
v_hc = zeros(3);

%% State 1 for agent 1 or agent 2
% In the first state, there are two variables: x1 = private good for agent
% 1, x2 = private good for agent 2
% public good expenditure equals the rest of the budget
c_hc(1,1) = x(1);
d_hc(1,1) = x(2);
Q_hc(1,1) = (y(1,1) - x(1) - x(2))./price;

% In second and third state, require allocation is at least as large as
% first state
x(3) = max(x(3),x(2));
x(4) = max(x(4),x(2));
x(5) = max(x(5),x(1));
x(6) = max(x(6),x(1));

% allocations
d_hc(2,1) = x(3);
d_hc(3,1) = x(4);
c_hc(1,2) = x(5);
c_hc(1,3) = x(6);

% agent 1: state 2/3, agent 2: state 1
Q_hc(1,2) = exp(log(Q_hc(1,1)) - (a1./(1-a1)).*log(c_hc(1,2)) + (a1./(1-a1)).*log(c_hc(1,1)+delta1.*rho));
Q_hc(1,3) = exp(log(Q_hc(1,2)) - (a1./(1-a1)).*log(c_hc(1,3)) + (a1./(1-a1)).*log(c_hc(1,2)+delta1.*rho));
d_hc(1,2) = y(1,2) - price*Q_hc(1,2) - c_hc(1,2);
d_hc(1,3) = y(1,3) - price*Q_hc(1,3) - c_hc(1,3);

Q_hc(2,1) = exp(log(Q_hc(1,1)) - (a2./(1-a2)).*log(d_hc(2,1)) + (a2./(1-a2)).*log(d_hc(1,1)+delta2));
Q_hc(3,1) = exp(log(Q_hc(2,1)) - (a2./(1-a2)).*log(d_hc(3,1)) + (a2./(1-a2)).*log(d_hc(2,1)+delta2));
c_hc(2,1) = y(2,1) - price*Q_hc(2,1) - d_hc(2,1);
c_hc(3,1) = y(3,1) - price*Q_hc(3,1) - d_hc(3,1);

% agent 1: state 2, agent 2: state 2
%fun = @(z) a1.*log(y(2,2)-exp((1-a2)*log(Q_hc(1,2))/a2 - (1-a2)*log(z)/a2 + log(d_hc(1,2)+delta2)) - price*z) - a1*log(c_hc(2,1)+delta1*rho) - (1-a1)*log(Q_hc(2,1)) + (1-a1)*z;
%Q_hc(2,2) = fzero(fun,2);
%d_hc(2,2) = exp((1-a2)*log(Q_hc(1,2))/a2 - (1-a2)*log(Q_hc(2,2))/a2 + log(d_hc(1,2)+delta2));
%c_hc(2,2) = y(2,2) - price*Q_hc(2,2) - d_hc(2,2);


%% States 2 and 3 for agents
% agent 1: state 2, agent 2: state 2
% hiding utility for agent 1
u_hc(2,2) = a1.*log(c_hc(2,1)+delta1*rho)+(1-a1).*log(Q_hc(2,1));
% hiding utility for agent 2
v_hc(2,2) = a2.*log(d_hc(1,2)+delta2)+(1-a2).*log(Q_hc(1,2)); 
% calculate allocation
startval = d_hc(1,2)+delta2;
[c_hc(2,2),d_hc(2,2),Q_hc(2,2)] = hold_u_v_constant(u_hc(2,2),v_hc(2,2),y(2,2),a1,a2,price,startval);

% agent 1: state 3, agent 2: state 2
% hiding utility for agent 1
u_hc(2,3) = a1.*log(c_hc(2,2)+delta1*rho)+(1-a1).*log(Q_hc(2,2));
% hiding utility for agent 2
v_hc(2,3) = a2.*log(d_hc(1,3)+delta2)+(1-a2).*log(Q_hc(1,3)); 
% calculate allocation
startval = d_hc(1,3)+delta2;
[c_hc(2,3),d_hc(2,3),Q_hc(2,3)] = hold_u_v_constant(u_hc(2,3),v_hc(2,3),y(2,3),a1,a2,price,startval);

% agent 1: state 2, agent 2: state 3
% hiding utility for agent 1
u_hc(3,2) = a1.*log(c_hc(3,1)+delta1*rho)+(1-a1).*log(Q_hc(3,1));
% hiding utility for agent 2
v_hc(3,2) = a2.*log(d_hc(2,2)+delta2)+(1-a2).*log(Q_hc(2,2)); 
% calculate allocation
startval = d_hc(2,2)+delta2;
[c_hc(3,2),d_hc(3,2),Q_hc(3,2)] = hold_u_v_constant(u_hc(3,2),v_hc(3,2),y(3,2),a1,a2,price,startval);

% agent 1: state 3, agent 2: state 3
% hiding utility for agent 1
u_hc(3,3) = a1.*log(c_hc(3,2)+delta1*rho)+(1-a1).*log(Q_hc(3,2));
% hiding utility for agent 2
v_hc(3,3) = a2.*log(d_hc(2,3)+delta2)+(1-a2).*log(Q_hc(2,3));
% calculate allocation
startval = d_hc(2,3)+delta2;
[c_hc(3,3),d_hc(3,3),Q_hc(3,3)] = hold_u_v_constant(u_hc(3,3),v_hc(3,3),y(3,3),a1,a2,price,startval);

%% Optimize
% Feasibility
udiff_1_round = round(a1.*log(c_hc)+(1-a1).*log(Q_hc)-u_hc,3);
udiff_2_round = round(a2.*log(d_hc)+(1-a2).*log(Q_hc)-v_hc,3);
udiff_1 = (udiff_1_round<0);
udiff_2 = (udiff_2_round<0);
agent1_infeasible = sum(sum(udiff_1(2:3,2:3)));
agent2_infeasible = sum(sum(udiff_2(2:3,2:3)));
% Calculate household utility
eu = -sum(sum(prob.*(alpha.*u_hc + (1-alpha).*v_hc)))+99999.*agent1_infeasible + 99999.*agent2_infeasible;

end
