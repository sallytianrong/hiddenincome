function [eu,c_hc,d_hc,Q_hc,infeasible] = honest_bilateral_public_good_feasible(z,a1,a2,delta1,delta2,rho,price,y,prob,alpha,c_hc,d_hc,Q_hc,u_hc,v_hc);

d_hc(1,2) = z(1);
d_hc(1,3) = z(2);
c_hc(2,1) = z(3);
c_hc(3,1) = z(4);

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

% feasibility check
udiff_1 = (a1.*log(c_hc)+(1-a1).*log(Q_hc)-u_hc<-0.0001);
udiff_2 = (a2.*log(d_hc)+(1-a2).*log(Q_hc)-v_hc<-0.0001);
agent1_infeasible = sum(sum(udiff_1(2:3,2:3)));
agent2_infeasible = sum(sum(udiff_2(2:3,2:3)));
infeasible = (agent1_infeasible + agent2_infeasible > 0);

%% Calculate utility
h_hc = alpha.*u_hc + (1-alpha).*v_hc;
% expected utility
eu = - sum(sum(prob.*h_hc))+99999.*agent1_infeasible + 99999.*agent2_infeasible;
