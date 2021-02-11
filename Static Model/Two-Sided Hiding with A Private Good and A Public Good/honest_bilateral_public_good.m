function [eu, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_bilateral_public_good(x,a1,a2,price,delta1,delta2,alpha,y,rho,prob)

% make sure starting point is within bounds
x = max([0.01;0.01],x);
x(2) = min(x(2),y(1,1)-x(1)-0.01);

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
% utility
u_hc(1,1) = a1.*log(c_hc(1,1))+(1-a1).*log(Q_hc(1,1));
v_hc(1,1) = a2.*log(d_hc(1,1))+(1-a2).*log(Q_hc(1,1));

% agent 1: state 2, agent 2: state 1
% hiding utility for agent 1
u_hc(1,2) = a1.*log(c_hc(1,1)+delta1*rho)+(1-a1).*log(Q_hc(1,1));
% calculate allocations
[c_hc(1,2), d_hc_max_2, Q_hc(1,2)] = hold_u_constant(1,u_hc(1,2),y(1,2),a1,a2,price);

% agent 1: state 3, agent 2: state 1
% hiding utility for agent 1
u_hc(1,3) = a1.*log(c_hc(1,2)+delta1*rho)+(1-a1).*log(Q_hc(1,2));
% calculate allocations
[c_hc(1,3), d_hc_max_3, Q_hc(1,3)] = hold_u_constant(1,u_hc(1,3),y(1,3),a1,a2,price);

% agent 1: state 1, agent 2: state 2
% hiding utility for agent 2
v_hc(2,1) = a2.*log(d_hc(1,1)+delta2)+(1-a2).*log(Q_hc(1,1));
% calculate allocations
[d_hc(2,1), c_hc_max_2, Q_hc(2,1)] = hold_u_constant(2,v_hc(2,1),y(2,1),a1,a2,price);

% agent 1: state 1, agent 2: state 3
% hiding utility for agent 2
v_hc(3,1) = a2.*log(d_hc(2,1)+delta2)+(1-a2).*log(Q_hc(2,1));
% calculate allocations
[d_hc(3,1), c_hc_max_3, Q_hc(3,1)] = hold_u_constant(2,v_hc(3,1),y(3,1),a1,a2,price);

%% Find feasible allocation over the rest of the states
% fmincon options
options = optimoptions('fmincon','Display','off');
feasible = @(z) honest_bilateral_public_good_feasible(z,a1,a2,delta1,delta2,rho,price,y,prob,alpha,c_hc,d_hc,Q_hc,u_hc,v_hc);
z = fmincon(feasible,[d_hc(1,1);d_hc(1,1);c_hc(1,1);c_hc(1,1)].*1.001,[],[],[],[],...
    [d_hc(1,1);d_hc(1,1);c_hc(1,1);c_hc(1,1)],[d_hc_max_2;d_hc_max_3;c_hc_max_2;c_hc_max_3],[],options);
[~,c_hc,d_hc,Q_hc,infeasible] = honest_bilateral_public_good_feasible(z,a1,a2,delta1,delta2,rho,price,y,prob,alpha,c_hc,d_hc,Q_hc,u_hc,v_hc);

% Calculate utility
u_hc = a1.*log(c_hc) + (1-a1).*log(Q_hc);
v_hc = a2.*log(d_hc) + (1-a2).*log(Q_hc);
% Calculate household utility
eu = -sum(sum(prob.*(alpha.*u_hc + (1-alpha).*v_hc))) + infeasible.*99999;

end
