function [c, d, Q, eu, ev, eh, esp, ems, u_share, v_share] = unilateral_hiding_public_good(a1,a2,price,delta,alpha,y0,rho,p)
% Unilateral Hiding with One Private Good and One Public Good
% July 3, 2020

%{
% cobb-douglas utility function parameter
% utility = a*log(private good) + (1-a)*log(public good)
a1 = 0.5;
a2 = 0.5;
% price for the public good (price of the private good is
% normalized to 1)
price = 1;
% cost of hiding for private good (cannot hide public good)
delta = 0.9;
% alpha [0,1] is the Pareto weight of agent 1.
alpha = 0.5;
% y is the income
y0 = 1;
rho = 1;
% p is the probability of states for agent 1
p = [0.33 0.33 0.33];
%}

% calculate income in all states
y1 = [y0;y0+rho;y0+2*rho];
y2 = [2;2;2];
y = y1 + y2;

%% No Marriage
% Each agent chooses their own consumption of private and public goods.
c_nm = a1.*y1;
d_nm = a2.*y2;
Q_nm = (1-a1).*y1 + (1-a2).*y2;
% utility
u_nm = a1.*log(c_nm) + (1-a1).*log((1-a1).*y1);
v_nm = a2.*log(d_nm) + (1-a2).*log((1-a2).*y2);
% household utility
h_nm = u_nm + v_nm;

%% First-best (no lying)
% private and public consumption
% c = agent 1, d = agent 2, Q = public good
% characteristics of Cobb Douglas is fixed share of expenditure on goods
c_fb = alpha.*a1.*y;
d_fb = (1-alpha).*a2.*y;
Q_fb = (1-alpha.*a1-(1-alpha).*a2).*y./price;
% utility
% u = agent 1, v = agent 2
u_fb = a1.*log(c_fb) + (1-a1).*log(Q_fb);
v_fb = a2.*log(d_fb) + (1-a2).*log(Q_fb);
% expected utility
eu_fb = p*u_fb;
ev_fb = p*v_fb;
% household utility
h_fb = alpha.*u_fb + (1-alpha).*v_fb;
eh_fb = p*h_fb;
% equally-weighted utility
sp_fb = 0.5.*u_fb + 0.5.*v_fb;
esp_fb = p*sp_fb;
% marriage surplus
ms_fb = u_fb + v_fb - h_nm;
ems_fb = p*ms_fb;
% marital surplus share
eu_ms_fb = p*(u_fb - u_nm);
ev_ms_fb = p*(v_fb - v_nm);
u_share_fb = eu_ms_fb./ems_fb;
v_share_fb = ev_ms_fb./ems_fb;

%% Honest equilibrium with compensation (satisfies IC constraints)
% Choose allocation in state 1 that maximizes expected household utility
honest = @(x) honest_public_good(x,a1,a2,price,delta,alpha,y,rho,p);
%options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
rng default % For reproducibility
ms = MultiStart;
problem = createOptimProblem('fmincon','x0',[c_fb(1);d_fb(1)],'objective',honest,'lb',[0;0],'ub',[y(1);y(1)],'Aineq',[1 1],'bineq',y(1));
[c,~,~,~,~] = run(ms,problem,3);
%c = fmincon(honest,[c_fb(1);d_fb(1)]);
[~, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_public_good(c,a1,a2,price,delta,alpha,y,rho,p);
% expected utility
eu_hc = p*u_hc;
ev_hc = p*v_hc;
% household utility
h_hc = alpha.*u_hc + (1-alpha).*v_hc;
eh_hc = p*h_hc;
% equally-weighted utility
sp_hc = 0.5.*u_hc + 0.5.*v_hc;
esp_hc = p*sp_hc;
% marriage surplus
ms_hc = u_hc + v_hc - h_nm;
ems_hc = p*ms_hc;
% marital surplus share
eu_ms_hc = p*(u_hc - u_nm);
ev_ms_hc = p*(v_hc - v_nm);
u_share_hc = eu_ms_hc./ems_hc;
v_share_hc = ev_ms_hc./ems_hc;

%% Dishonest equilibrium: Agent 1 lies
% According to Munro, agent 1 cannot lie for the lowest and highest states.
% Therefore, agent 1 can only lie in state 2.
c_de = c_fb;
c_de(2) = c_fb(1)+delta*rho;
d_de = d_fb;
d_de(2) = d_fb(1);
Q_de = Q_fb;
Q_de(2) = Q_fb(1);
% utility
u_de = u_fb;
u_de(2) = a1.*log(c_fb(1)+delta*rho) + (1-a1).*log(Q_fb(1));
v_de = v_fb;
v_de(2) = v_fb(1);
% expected utility
eu_de = p*u_de;
ev_de = p*v_de;
% household utility
h_de = alpha.*u_de + (1-alpha).*v_de;
eh_de = p*h_de;
% equally-weighted utility
sp_de = 0.5.*u_de + 0.5.*v_de;
esp_de = p*sp_de;
% marriage surplus
ms_de = u_de + v_de - h_nm;
ems_de = p*ms_de;
% marital surplus share
eu_ms_de = p*(u_de - u_nm);
ev_ms_de = p*(v_de - v_nm);
u_share_de = eu_ms_de./ems_de;
v_share_de = ev_ms_de./ems_de;

%% Results
% all utilities in one matrix
c = [c_fb c_hc c_de];
d = [d_fb d_hc d_de];
Q = [Q_fb Q_hc Q_de];
eu = [eu_fb eu_hc eu_de];
ev = [ev_fb ev_hc ev_de];
eh = [eh_fb eh_hc eh_de];
esp = [esp_fb esp_hc esp_de];
ems = [ems_fb ems_hc ems_de];
u_share = [u_share_fb u_share_hc u_share_de];
v_share = [v_share_fb v_share_hc v_share_de];
