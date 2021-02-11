function [u, eu, v, ev, h, eh] = unilateral_hiding(delta, alpha, y0, p) 
% Unilateral Income Hiding: This function inputs delta (cost of hiding),
% alpha (Pareto weights), y0 (income) and p (probability of states) and
% outputs agents' and household's expected utility. u is the utility of
% agent 1 (who can hide) and v is the utility of agent 2. The utility
% function is u = sqrt(c). There are three states of the world and income
% of agent 2 is normalized to a constant 0 across all states.
% Last modified: April 16, 2020

%% Set up
%{
% Uncomment to run as a script rather than a function
% Parameters
% delta [0,1] is cost of hiding. delta=1 means there is no cost of hiding.
% delta=0 means cost of hiding is prohibitive.
delta = 0.9;
% alpha [0,1] is the Pareto weight of agent 1.
alpha = 0.9;
% y is the income
y0 = 5;
% p is the probability of states
p = [0.33 0.33 0.33];
%}
% Calculate income in all states.
y = [y0;y0+1;y0+2];

%% First-best (no lying)
% consumption
alpha_con = alpha.^2 + (1-alpha).^2;
c1_fb = alpha.^2.*y./alpha_con;
c2_fb = (1-alpha).^2.*y./alpha_con;
% resource shares
rs1_fb = c1_fb./(c1_fb+c2_fb);
rs2_fb = c2_fb./(c1_fb+c2_fb);
% utility
u_fb = c1_fb.^0.5;
v_fb = c2_fb.^0.5;
% expected utility
eu_fb = p*u_fb;
ev_fb = p*v_fb;
% household utility
h_fb = alpha.*u_fb + (1-alpha).*v_fb;
eh_fb = p*h_fb;

%% Honest equilibrium with compensation (satisfies IC constraints)
fun = @(x) -alpha.*(p(1).*x.^0.5 + p(2).*(x+delta).^0.5 + p(3).*(x+2.*delta).^0.5)...
    - (1-alpha).*(p(1).*(y(1)-x).^0.5 + p(2).*(y(2)-x-delta).^0.5 + p(3).*(y(3)-x-2.*delta).^0.5);
%fun = @(x) alpha.*(p(1)./x.^0.5 + p(2)./(x+delta).^0.5 + p(3)./(x+2.*delta).^0.5)...
%    - (1-alpha).*(p(1)./(y(1)-x).^0.5 + p(2)./(y(2)-x-delta).^0.5 + p(3)./(y(3)-x-2.*delta).^0.5);
x0 = 3.5;
c = fmincon(fun,x0,[],[],[],[],0,y(1));
% consumption
c1_hc = [c;c+delta;c+2.*delta];
c2_hc = y - c1_hc;
% resource shares
rs1_hc = c1_hc./(c1_hc+c2_hc);
rs2_hc = c2_hc./(c1_hc+c2_hc);
% utility
u_hc = c1_hc.^0.5;
v_hc = c2_hc.^0.5;
% expected utility
eu_hc = p*u_hc;
ev_hc = p*v_hc;
% household utility
h_hc = alpha.*u_hc + (1-alpha).*v_hc;
eh_hc = p*h_hc;

%% The dishonest equilibrium
% According to Munro, agent 1 cannot lie for the lowest and highest states.
% Therefore, agent 1 can only lie in state 2.

% consumption
c1_de = [c1_fb(1);c1_fb(1)+delta;c1_fb(3)];
c2_de = [c2_fb(1);c2_fb(1);c2_fb(3)];
% resource shares
rs1_de = c1_de./(c1_de+c2_de);
rs2_de = c2_de./(c1_de+c2_de);
% utility
u_de = c1_de.^0.5;
v_de = c2_de.^0.5;
% expected utility
eu_de = p*u_de;
ev_de = p*v_de;
% household utility
h_de = alpha.*u_de + (1-alpha).*v_de;
eh_de = p*h_de;

%% Results
% all utilities in one matrix
u = [u_fb u_hc u_de];
c1 = [c1_fb c1_hc c1_de];
rs1 = [rs1_fb rs1_hc rs1_de];
eu = [eu_fb eu_hc eu_de];
v = [v_fb v_hc v_de];
c2 = [c2_fb c2_hc c2_de];
rs2 = [rs2_fb rs2_hc rs2_de];
ev = [ev_fb ev_hc ev_de];
h = [h_fb h_hc h_de];
eh = [eh_fb eh_hc eh_de];

