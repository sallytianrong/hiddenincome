function [u, eu, ev, eh, esp, ems, eu_ms, ev_ms] = bilateral_hiding_log(delta1, delta2, alpha, y0, rho, p, q) 
% Bilateral Income Hiding with log utility
% April 23, 2020
% April 23: add social planner utility where Pareto weight of each agent is
% 0.5

%% Set up
%{
% Uncomment this to run as a script
% Parameters
% delta1, delta2 [0,1] is cost of hiding. delta=1 means there is no cost of hiding.
% delta=0 means cost of hiding is prohibitive. delta1 is the cost of hiding
% for agent 1, delta2 is the cost of hiding for agent 2.
delta1 = 0.9;
delta2 = 0.9;
% alpha [0,1] is the Pareto weight of agent 1.
alpha = 0.6;
% Agent 2's income is normalized to 0,1,2 in the three states. Agent 1's
% income are y0, y0+rho, y0+2rho.
y0 = 3;
rho = 1;
% p is the probability of states for agent 1, q is the probability of
% states for agent 2;
p = [0.33 0.33 0.33];
q = [0.33;0.33;0.33];
%}

% Calculate income in all states and probability of states
y = [y0 y0+rho y0+2*rho];
x = [1;2;3];
inc = y+x;
prob = p.*q;

%% No Marriage
% Each agent consumes their own income.
u_nm = repmat(log(y),3,1);
v_nm = repmat(log(x),1,3);
% expected utility
eu_nm = sum(sum(prob.*u_nm));
ev_nm = sum(sum(prob.*v_nm));
% household utility
h_nm = u_nm + v_nm;
eh_nm = sum(sum(prob.*h_nm));
% equally-weighted utility
sp_nm = 0.5.*u_nm + 0.5.*v_nm;
esp_nm = sum(sum(prob.*sp_nm));

%% First-best (no lying)
% consumption
c1_fb = alpha.*inc;
c2_fb = (1-alpha).*inc;
% utility
u_fb = log(c1_fb);
v_fb = log(c2_fb);
% resource shares
rs1_fb = c1_fb./(c1_fb+c2_fb);
rs2_fb = c2_fb./(c1_fb+c2_fb);
ers1_fb = sum(sum(prob.*rs1_fb));
ers2_fb = sum(sum(prob.*rs2_fb));
% expected utility
eu_fb = sum(sum(prob.*u_fb));
ev_fb = sum(sum(prob.*v_fb));
% household utility
h_fb = alpha.*u_fb + (1-alpha).*v_fb;
eh_fb = sum(sum(prob.*h_fb));
% social planner utility
sp_fb = 0.5.*u_fb + 0.5.*v_fb;
esp_fb = sum(sum(prob.*sp_fb));
% marriage surplus
ms_fb = u_fb + v_fb - h_nm;
ems_fb = sum(sum(prob.*ms_fb));
% marital surplus share
eu_ms_fb = sum(sum(prob.*(u_fb - u_nm)));
ev_ms_fb = sum(sum(prob.*(v_fb - v_nm)));
u_share_fb = eu_ms_fb./ems_fb;
v_share_fb = ev_ms_fb./ems_fb;

%% Incentive compatible equilibrium
% first check whether first-best is incentive compatible
ic1 = (c1_fb(:,2)-c1_fb(:,1)-delta1<0);
ic2 = (c1_fb(:,3)-c1_fb(:,2)-delta1<0);
ic3 = (c2_fb(2,:)-c2_fb(1,:)-delta2<0);
ic4 = (c2_fb(3,:)-c2_fb(2,:)-delta2<0);
tot_ic = sum(ic1) + sum(ic2) + sum(ic3) + sum(ic4);
if tot_ic == 0
    c1_hc = c1_fb;
    c2_hc = c2_fb;
    rs1_hc = rs1_fb;
    rs2_hc = rs2_fb;
    ers1_hc = ers1_fb;
    ers2_hc = ers2_fb;
    u_hc = u_fb;
    v_hc = v_fb;
    eu_hc = eu_fb;
    ev_hc = ev_fb;
    h_hc = h_fb;
    eh_hc = eh_fb;
    sp_hc = sp_fb;
    esp_hc = esp_fb;
else
    % if first-best is not incentive compatible, resolve optimization problem
    % with incentive compatibility constraints
    fun = @(x) honest_log(x,alpha,delta1,delta2,inc,rho,prob);
    x0 = c1_fb(1,1);
    c = fmincon(fun,x0,[],[],[],[],0,y0+1);
    % consumption
    c1_hc = zeros(size(prob));
    c1_hc(:,1) = [c;c+1-delta2;c+2-2*delta2];
    c1_hc =[c1_hc(:,1) c1_hc(:,1)+delta1.*rho c1_hc(:,1)+2*delta1.*rho];
    c2_hc = inc - c1_hc;
    % resource shares
    rs1_hc = c1_hc./(c1_hc+c2_hc);
    rs2_hc = c2_hc./(c1_hc+c2_hc);
    ers1_hc = sum(sum(prob.*rs1_hc));
    ers2_hc = sum(sum(prob.*rs2_hc));
    % utility
    u_hc = log(c1_hc);
    v_hc = log(c2_hc);
    % expected utility
    eu_hc = sum(sum(prob.*u_hc));
    ev_hc = sum(sum(prob.*v_hc));
    % household utility
    h_hc = alpha.*u_hc + (1-alpha).*v_hc;
    eh_hc = sum(sum(prob.*h_hc));
    % social planner utility
    sp_hc = 0.5.*u_hc + 0.5.*v_hc;
    esp_hc = sum(sum(prob.*sp_hc));
end
% marriage surplus
ms_hc = u_hc + v_hc - h_nm;
ems_hc = sum(sum(prob.*ms_hc));
% marital surplus share
eu_ms_hc = sum(sum(prob.*(u_hc - u_nm)));
ev_ms_hc = sum(sum(prob.*(v_hc - v_nm)));
u_share_hc = eu_ms_hc./ems_hc;
v_share_hc = ev_ms_hc./ems_hc;

%% Dishonest equilibrium 1: Agent 1 lies, Agent 2 does not lie
% According to Munro, agent 1 cannot lie for the lowest and highest states.
% Therefore, agent 1 can only lie in state 2.
% consumption
c1_de1 = [c1_fb(:,1) c1_fb(:,1)+delta1*rho c1_fb(:,3)];
c2_de1 = [c2_fb(:,1) c2_fb(:,1) c2_fb(:,3)];
% resource shares
rs2_de1 = c2_de1./inc;
rs1_de1 = ones(3) - rs2_de1;
ers1_de1 = sum(sum(prob.*rs1_de1));
ers2_de1 = sum(sum(prob.*rs2_de1));
% utility
u_de1 = log(c1_de1);
v_de1 = log(c2_de1);
% expected utility
eu_de1 = sum(sum(prob.*u_de1));
ev_de1 = sum(sum(prob.*v_de1));
% household utility
h_de1 = alpha.*u_de1 + (1-alpha).*v_de1;
eh_de1 = sum(sum(prob.*h_de1));
% social planner utility
sp_de1 = 0.5.*u_de1 + 0.5.*v_de1;
esp_de1 = sum(sum(prob.*sp_de1));
% marriage surplus
ms_de1 = u_de1 + v_de1 - h_nm;
ems_de1 = sum(sum(prob.*ms_de1));
% marital surplus share
eu_ms_de1 = sum(sum(prob.*(u_de1 - u_nm)));
ev_ms_de1 = sum(sum(prob.*(v_de1 - v_nm)));
u_share_de1 = eu_ms_de1./ems_de1;
v_share_de1 = ev_ms_de1./ems_de1;

%% Dishonest equilibrium 2: Agent 2 lies, Agent 1 does not lie
% According to Munro, agent 2 cannot lie for the lowest and highest states.
% Therefore, agent 2 can only lie in state 2.
% consumption
c1_de2 = [c1_fb(1,:);c1_fb(1,:);c1_fb(3,:)];
c2_de2 = [c2_fb(1,:);c2_fb(1,:)+delta2;c2_fb(3,:)];
% resource shares
rs1_de2 = c1_de2./inc;
rs2_de2 = ones(3) - rs1_de2;
ers1_de2 = sum(sum(prob.*rs1_de2));
ers2_de2 = sum(sum(prob.*rs2_de2));
% utility
u_de2 = log(c1_de2);
v_de2 = log(c2_de2);
% expected utility
eu_de2 = sum(sum(prob.*u_de2));
ev_de2 = sum(sum(prob.*v_de2));
% household utility
h_de2 = alpha.*u_de2 + (1-alpha).*v_de2;
eh_de2 = sum(sum(prob.*h_de2));
% social planner utility
sp_de2 = 0.5.*u_de2 + 0.5.*v_de2;
esp_de2 = sum(sum(prob.*sp_de2));
% marriage surplus
ms_de2 = u_de2 + v_de2 - h_nm;
ems_de2 = sum(sum(prob.*ms_de2));
% marital surplus share
eu_ms_de2 = sum(sum(prob.*(u_de2 - u_nm)));
ev_ms_de2 = sum(sum(prob.*(v_de2 - v_nm)));
u_share_de2 = eu_ms_de2./ems_de2;
v_share_de2 = ev_ms_de2./ems_de2;

%% Dishonest equilibrium 3: Both lie
% According to Munro, agents cannot lie for the lowest and highest states.
% Therefore, agents can only lie in state 2.
% consumption
c1_de3 = [c1_de2(:,1) c1_de2(:,1)+delta1*rho c1_de2(:,3)];
c2_de3 = [c2_de1(1,:);c2_de1(1,:)+delta2;c2_de1(3,:)];
% resource shares
c1_de3_rs = [c1_de2(:,1) c1_de2(:,1)+rho c1_de2(:,3)];
c2_de3_rs = [c2_de1(1,:);c2_de1(1,:)+1;c2_de1(3,:)];
rs1_de3 = c1_de3_rs./(c1_de3_rs+c2_de3_rs);
rs2_de3 = c2_de3_rs./(c1_de3_rs+c2_de3_rs);
ers1_de3 = sum(sum(prob.*rs1_de3));
ers2_de3 = sum(sum(prob.*rs2_de3));
% utility
u_de3 = log(c1_de3);
v_de3 = log(c2_de3);
% expected utility
eu_de3 = sum(sum(prob.*u_de3));
ev_de3 = sum(sum(prob.*v_de3));
% household utility
h_de3 = alpha.*u_de3 + (1-alpha).*v_de3;
eh_de3 = sum(sum(prob.*h_de3));
% social planner utility
sp_de3 = 0.5.*u_de3 + 0.5.*v_de3;
esp_de3 = sum(sum(prob.*sp_de3));
% marriage surplus
ms_de3 = u_de3 + v_de3 - h_nm;
ems_de3 = sum(sum(prob.*ms_de3));
% marital surplus share
eu_ms_de3 = sum(sum(prob.*(u_de3 - u_nm)));
ev_ms_de3 = sum(sum(prob.*(v_de3 - v_nm)));
u_share_de3 = eu_ms_de3./ems_de3;
v_share_de3 = ev_ms_de3./ems_de3;

%% Results
% all utilities in one matrix
u = cat(3,u_fb,u_hc,u_de1,u_de2,u_de3);
c1 = cat(3,c1_fb,c1_hc,c1_de1,c1_de2,c1_de3);
rs1 = cat(3,rs1_fb,rs1_hc,rs1_de1,rs1_de2,rs1_de3);
eu = [eu_fb eu_hc eu_de1 eu_de2 eu_de3];
ers1 = [ers1_fb ers1_hc ers1_de1 ers1_de2 ers1_de3];

v = cat(3,v_fb,v_hc,v_de1,v_de2,v_de3);
c2 = cat(3,c2_fb,c2_hc,c2_de1,c2_de2,c2_de3);
rs2 = cat(3,rs2_fb,rs2_hc,rs2_de1,rs2_de2,rs2_de3);
ev = [ev_fb ev_hc ev_de1 ev_de2 ev_de3];
ers2 = [ers2_fb ers2_hc ers2_de1 ers2_de2 ers2_de3];

h = cat(3,h_fb,h_hc,h_de1,h_de2,h_de3);
eh = [eh_fb eh_hc eh_de1 eh_de2 eh_de3];

sp = cat(3,sp_fb,sp_hc,sp_de1,sp_de2,sp_de3);
esp = [esp_fb esp_hc esp_de1 esp_de2 esp_de3];

ems = [ems_fb ems_hc ems_de1 ems_de2 ems_de3];
eu_ms = [eu_ms_fb eu_ms_hc eu_ms_de1 eu_ms_de2 eu_ms_de3];
ev_ms = [ev_ms_fb ev_ms_hc ev_ms_de1 ev_ms_de2 ev_ms_de3];
u_share = [u_share_fb u_share_hc u_share_de1 u_share_de2 u_share_de3];
v_share = [v_share_fb v_share_hc v_share_de1 v_share_de2 v_share_de3];

c_tot = c1+c2;

end
