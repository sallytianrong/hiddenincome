function [eco, ecu, edo, edu, etoto, etotu, eu, ev, eh, esp] = bilateral_hiding_twogoods(a1, a2, price, delta1o, delta1u, delta2o, delta2u, alpha, y0, rho, p, q) 
% Bilateral Hiding with Observable and Unobservable Consumption
% Introducing observable and unobservable goods. For right now there is
% only one observable good and one unobservable.
% May 13, 2020

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
delta1u = 0.9;
delta2u = 0.9;
% alpha [0,1] is the Pareto weight of agent 1.
alpha = 0.3;
% Agent 2's income is normalized to 0,1,2 in the three states. Agent 1's
% income are y0, y0+rho, y0+2rho.
y0 = 2;
rho = 1;
% p is the probability of states for agent 1, q is the probability of
% states for agent 2;
p = [0.33 0.33 0.33];
q = [0.33;0.33;0.33];
%}

% Calculate income in all states and probability of states
y1 = [y0 y0+rho y0+2*rho];
y2 = [1;2;3];
inc = y1+y2;
prob = p.*q;

%% First-best (no lying)
% observable and unobservable consumption
% c = agent 1, d = agent 2, o = observable, u = unobservable
% characteristics of Cobb Douglas is fixed share of expenditure on goods
co_fb = a1.*alpha.*inc;
cu_fb = (1-a1).*alpha.*inc./price;
do_fb = a2.*(1-alpha).*inc;
du_fb = (1-a2).*(1-alpha).*inc./price;
% utility
% u = agent 1, v = agent 2
u_fb = a1.*log(co_fb) + (1-a1).*log(cu_fb);
v_fb = a2.*log(do_fb) + (1-a2).*log(du_fb);
% expected utility
eu_fb = sum(sum(prob.*u_fb));
ev_fb = sum(sum(prob.*v_fb));
% household utility
h_fb = alpha.*u_fb + (1-alpha).*v_fb;
eh_fb = sum(sum(prob.*h_fb));
% equally-weighted utility
sp_fb = 0.5.*u_fb + 0.5.*v_fb;
esp_fb = sum(sum(prob.*sp_fb));

%% Honest equilibrium with compensation (satisfies IC constraints)
% Find optimal resource allocation in state (1,1)
fun = @(x) honest_twogoods(x, a1, a2, rho, price, delta1o, delta1u, delta2o, delta2u, inc, alpha, prob);
x0 = [alpha.*inc(1,1);(1-alpha).*inc(1,1)];
options = optimoptions('fmincon','Display','Off');
x1 = fmincon(fun,x0,[1 1],inc(1,1),[],[],[0;0],[],[],options);
% Calculate optimal resource allocation in all states
[~, res1, res2, alloc_diff] = honest_twogoods(x1, a1, a2, rho, price, delta1o, delta1u, delta2o, delta2u, inc, alpha, prob);
alloc_diff
% Calculate consumption
co_hc = a1.*res1;
cu_hc = (1-a1).*res1./price;
do_hc = a2.*res2;
du_hc = (1-a2).*res2./price;
% Calculate utility in all states
u_hc = a1.*log(co_hc) + (1-a1).*log(cu_hc);
v_hc = a2.*log(do_hc) + (1-a2).*log(du_hc);
% expected utility
eu_hc = sum(sum(prob.*u_hc));
ev_hc = sum(sum(prob.*v_hc));
% household utility
h_hc = alpha.*u_hc + (1-alpha).*v_hc;
eh_hc = sum(sum(prob.*h_hc));
% equally-weighted utility
sp_hc = 0.5.*u_hc + 0.5.*v_hc;
esp_hc = sum(sum(prob.*sp_hc));

%% Dishonest equilibrium 1: Agent 1 lies, Agent 2 does not lie
% According to Munro, agent 1 cannot lie for the lowest and highest states.
% Therefore, agent 1 can only lie in state 2.

% For agent 2, utility is the same as first-best, with the exception that
% state 2 for agent 1 is regarded as state 1
do_de1 = [do_fb(:,1) do_fb(:,1) do_fb(:,3)];
du_de1 = [du_fb(:,1) du_fb(:,1) du_fb(:,3)];
v_de1 = [v_fb(:,1) v_fb(:,1) v_fb(:,3)];
ev_de1 = sum(sum(prob.*v_de1));

% consumption of agent 1 is the same as first-best under states 1 and 3
co_de1 = co_fb;
cu_de1 = cu_fb;

% In state 2 for agent 1, find agent 1's max utility through hiding
% state 1 for agent 2
u_hiding = @(x) besthiding(x, a1, alpha.*inc(1,1), rho, price, delta1o, delta1u);
init = rho.*(1-a1).*delta1u./price;
x_hiding = fmincon(u_hiding,init,[],[],[],[],0,rho.*delta1u./price,[],options);
[~, c_o_1, c_o_2, c_u_1, c_u_2] = besthiding(x_hiding, a1, alpha.*inc(1,1), rho, price, delta1o, delta1u);
co_de1(1,2) = c_o_1+c_o_2;
cu_de1(1,2) = c_u_1+c_u_2;

%%
% state 2 for agent 2
u_hiding = @(x) besthiding(x, a1, alpha.*inc(2,1), rho, price, delta1o, delta1u);
init = rho.*(1-a1).*delta1u./price;
x_hiding = fmincon(u_hiding,init,[],[],[],[],0,rho.*delta1u./price,[],options);
[~, c_o_1, c_o_2, c_u_1, c_u_2] = besthiding(x_hiding, a1, alpha.*inc(2,1), rho, price, delta1o, delta1u);
co_de1(2,2) = c_o_1+c_o_2;
cu_de1(2,2) = c_u_1+c_u_2;
% state 3 for agent 2
u_hiding = @(x) besthiding(x, a1, alpha.*inc(3,1), rho, price, delta1o, delta1u);
init = rho.*(1-a1).*delta1u./price;
x_hiding = fmincon(u_hiding,init,[],[],[],[],0,rho.*delta1u./price,[],options);
[~, c_o_1, c_o_2, c_u_1, c_u_2] = besthiding(x_hiding, a1, alpha.*inc(3,1), rho, price, delta1o, delta1u);
co_de1(3,2) = c_o_1+c_o_2;
cu_de1(3,2) = c_u_1+c_u_2;
% utility
u_de1 = a1.*log(co_de1) + (1-a1).*log(cu_de1);
eu_de1 = sum(sum(prob.*u_de1));

% household utility
h_de1 = alpha.*u_de1 + (1-alpha).*v_de1;
eh_de1 = sum(sum(prob.*h_de1));
% social planner utility
sp_de1 = 0.5.*u_de1 + 0.5.*v_de1;
esp_de1 = sum(sum(prob.*sp_de1));

%% Dishonest equilibrium 2: Agent 2 lies, Agent 1 does not lie
% According to Munro, agent 2 cannot lie for the lowest and highest states.
% Therefore, agent 2 can only lie in state 1.

% For agent 1, utility is the same as first-best, with the exception that
% state 2 for agent 2 is regarded as state 1
co_de2 = [co_fb(1,:);co_fb(1,:);co_fb(3,:)];
cu_de2 = [cu_fb(1,:);cu_fb(1,:);cu_fb(3,:)];
u_de2 = [u_fb(1,:);u_fb(1,:);u_fb(3,:)];
eu_de2 = sum(sum(prob.*u_de2));

% consumption of agent 2 is the same as first-best under states 1 and 3
do_de2 = do_fb;
du_de2 = du_fb;

% In state 2 for agent 2, find agent 2's max utility through hiding
% state 1 for agent 1
u_hiding = @(x) besthiding(x, a2, (1-alpha).*inc(1,1), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
x_hiding = fmincon(u_hiding,init,[],[],[],[],0,delta2u./price,[],options);
[~, c_o_1, c_o_2, c_u_1, c_u_2] = besthiding(x_hiding, a2, (1-alpha).*inc(1,1), 1, price, delta2o, delta2u);
do_de2(2,1) = c_o_1+c_o_2;
du_de2(2,1) = c_u_1+c_u_2;
% state 2 for agent 1
u_hiding = @(x) besthiding(x, a2, (1-alpha).*inc(1,2), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
x_hiding = fmincon(u_hiding,init,[],[],[],[],0,delta2u./price,[],options);
[~, c_o_1, c_o_2, c_u_1, c_u_2] = besthiding(x_hiding, a2, (1-alpha).*inc(1,2), 1, price, delta2o, delta2u);
do_de2(2,2) = c_o_1+c_o_2;
du_de2(2,2) = c_u_1+c_u_2;
% state 3 for agent 2
u_hiding = @(x) besthiding(x, a2, (1-alpha).*inc(1,3), 1, price, delta2o, delta2u);
init = (1-a2).*delta2u./price;
x_hiding = fmincon(u_hiding,init,[],[],[],[],0,delta2u./price,[],options);
[~, c_o_1, c_o_2, c_u_1, c_u_2] = besthiding(x_hiding, a2, (1-alpha).*inc(1,3), 1, price, delta2o, delta2u);
do_de2(2,3) = c_o_1+c_o_2;
du_de2(2,3) = c_u_1+c_u_2;
% utility
v_de2 = a1.*log(do_de2) + (1-a1).*log(du_de2);
ev_de2 = sum(sum(prob.*v_de2));

% household utility
h_de2 = alpha.*u_de1 + (1-alpha).*v_de1;
eh_de2 = sum(sum(prob.*h_de2));
% social planner utility
sp_de2 = 0.5.*u_de2 + 0.5.*v_de2;
esp_de2 = sum(sum(prob.*sp_de2));

%% Dishonest equilibrium 3: Both lie
% According to Munro, agents cannot lie for the lowest and highest states.
% Therefore, agents can only lie in state 2.

% this equilibrium is a combination of dishonest equilibria 1 and 2
co_de3 = [co_de1(1,:);co_de1(1,:);co_de1(3,:)];
cu_de3 = [cu_de1(1,:);cu_de1(1,:);cu_de1(3,:)];
do_de3 = [do_de2(:,1) do_de2(:,1) do_de2(:,3)];
du_de3 = [du_de2(:,1) du_de2(:,1) du_de2(:,3)];
% utility
u_de3 = a1.*log(co_de3) + (1-a1).*log(cu_de3);
v_de3 = a2.*log(do_de3) + (1-a2).*log(du_de3);
% expected utility
eu_de3 = sum(sum(prob.*u_de3));
ev_de3 = sum(sum(prob.*v_de3));
% household utility
h_de3 = alpha.*u_de3 + (1-alpha).*v_de3;
eh_de3 = sum(sum(prob.*h_de3));
% social planner utility
sp_de3 = 0.5.*u_de3 + 0.5.*v_de3;
esp_de3 = sum(sum(prob.*sp_de3));

%% Results
% all utilities in one matrix
co = cat(3,co_fb,co_hc,co_de1,co_de2,co_de3);
cu = cat(3,cu_fb,cu_hc,cu_de1,cu_de2,cu_de3);
eco = permute(sum(sum(prob.*co)),[1 3 2]);
ecu = permute(sum(sum(prob.*cu)),[1 3 2]);

do = cat(3,do_fb,do_hc,do_de1,do_de2,do_de3);
du = cat(3,du_fb,du_hc,du_de1,du_de2,du_de3);
edo = permute(sum(sum(prob.*do)),[1 3 2]);
edu = permute(sum(sum(prob.*du)),[1 3 2]);

toto = co+do;
totu = cu+du;
etoto = permute(sum(sum(prob.*toto)),[1 3 2]);
etotu = permute(sum(sum(prob.*totu)),[1 3 2]);

u = cat(3,u_fb,u_hc,u_de1,u_de2,u_de3);
eu = [eu_fb eu_hc eu_de1 eu_de2 eu_de3];

v = cat(3,v_fb,v_hc,v_de1,v_de2,v_de3);
ev = [ev_fb ev_hc ev_de1 ev_de2 ev_de3];

h = cat(3,h_fb,h_hc,h_de1,h_de2,h_de3);
eh = [eh_fb eh_hc eh_de1 eh_de2 eh_de3];

sp = cat(3,sp_fb,sp_hc,sp_de1,sp_de2,sp_de3);
esp = [esp_fb esp_hc esp_de1 esp_de2 esp_de3];

% total consumption
c_tot = co + do + price.*cu + price.*du;

end