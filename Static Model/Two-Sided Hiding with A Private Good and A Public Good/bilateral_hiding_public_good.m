function [eu, ev, eh, esp, ems, u_share, v_share, Q] = bilateral_hiding_public_good(a1,a2,price,delta1,delta2,alpha,y0,rho,p,q)
% Bilateral Hiding with One Private Good and One Public Good
% July 20, 2020

%{
% cobb-douglas utility function parameter
% utility = a*log(private good) + (1-a)*log(public good)
a1 = 0.5;
a2 = 0.5;
% price for the public good (price of the private good is
% normalized to 1)
price = 1;
% cost of hiding for private good for both agents (cannot hide public good)
delta1 = 0.2172;
delta2 = 0.8;
% alpha [0,1] is the Pareto weight of agent 1.
alpha = 0.7;
% y is the income
y0 = 1;
rho = 1;
% p is the probability of states for agent 1, q is the probability of
% states for agent 2
p = [0.33 0.33 0.33];
q = [0.33;0.33;0.33];
%}

% calculate income in all states
y1 = [y0 y0+rho y0+2*rho];
y2 = [1;2;3];
y = y1 + y2;
prob = p.*q;

%% No Marriage
% Each agent chooses their own consumption of private and public goods.
c_nm = a1.*y1;
d_nm = a2.*y2;
Q_nm = (1-a1).*y1 + (1-a2).*y2;
eQ_nm = sum(sum(prob.*Q_nm));
% utility
u_nm = a1.*log(c_nm) + (1-a1).*log((1-a1).*y1);
v_nm = a2.*log(d_nm) + (1-a2).*log((1-a2).*y2);
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
% private and public consumption
% c = agent 1, d = agent 2, Q = public good
% characteristics of Cobb Douglas is fixed share of expenditure on goods
c_fb = alpha.*a1.*y;
d_fb = (1-alpha).*a2.*y;
Q_fb = (1-alpha.*a1-(1-alpha).*a2).*y./price;
eQ_fb = sum(sum(prob.*Q_fb));
% utility
% u = agent 1, v = agent 2
u_fb = a1.*log(c_fb) + (1-a1).*log(Q_fb);
v_fb = a2.*log(d_fb) + (1-a2).*log(Q_fb);
% expected utility
eu_fb = sum(sum(prob.*u_fb));
ev_fb = sum(sum(prob.*v_fb));
% household utility
h_fb = alpha.*u_fb + (1-alpha).*v_fb;
eh_fb = sum(sum(prob.*h_fb));
% equally-weighted utility
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

%% Dishonest equilibrium: Agent 1 lies
% According to Munro, agent 1 cannot lie for the lowest and highest states.
% Therefore, agent 1 can only lie in state 2.
c_de1 = c_fb;
c_de1(:,2) = c_fb(:,1)+delta1*rho;
d_de1 = d_fb;
d_de1(:,2) = d_fb(:,1);
Q_de1 = Q_fb;
Q_de1(:,2) = Q_fb(:,1);
eQ_de1 = sum(sum(prob.*Q_de1));
% utility
u_de1 = u_fb;
u_de1(:,2) = a1.*log(c_fb(:,1)+delta1*rho) + (1-a1).*log(Q_fb(:,1));
v_de1 = v_fb;
v_de1(:,2) = v_fb(:,1);
% expected utility
eu_de1 = sum(sum(prob.*u_de1));
ev_de1 = sum(sum(prob.*v_de1));
% household utility
h_de1 = alpha.*u_de1 + (1-alpha).*v_de1;
eh_de1 = sum(sum(prob.*h_de1));
% equally-weighted utility
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

%% Dishonest equilibrium: Agent 2 lies
% According to Munro, agent 2 cannot lie for the lowest and highest states.
% Therefore, agent 2 can only lie in state 2.
c_de2 = c_fb;
c_de2(2,:) = c_fb(1,:);
d_de2 = d_fb;
d_de2(2,:) = d_fb(1,:)+delta2;
Q_de2 = Q_fb;
Q_de2(2,:) = Q_fb(1,:);
eQ_de2 = sum(sum(prob.*Q_de2));
% utility
u_de2 = a1.*log(c_de2) + (1-a1).*log(Q_de2);
v_de2 = a2.*log(d_de2) + (1-a2).*log(Q_de2);
% expected utility
eu_de2 = sum(sum(prob.*u_de2));
ev_de2 = sum(sum(prob.*v_de2));
% household utility
h_de2 = alpha.*u_de2 + (1-alpha).*v_de2;
eh_de2 = sum(sum(prob.*h_de2));
% equally-weighted utility
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

% this equilibrium is a combination of dishonest equilibria 1 and 2
c_de3 = [c_de1(1,:);c_de1(1,:);c_de1(3,:)];
d_de3 = [d_de2(:,1) d_de2(:,1) d_de2(:,3)];
Q_de3 = [Q_de1(1,:);Q_de1(1,:);Q_de1(3,:)];
eQ_de3 = sum(sum(prob.*Q_de3));
% utility
u_de3 = a1.*log(c_de3) + (1-a1).*log(Q_de3);
v_de3 = a2.*log(d_de3) + (1-a2).*log(Q_de3);
% expected utility
eu_de3 = sum(sum(prob.*u_de3));
ev_de3 = sum(sum(prob.*v_de3));
% household utility
h_de3 = alpha.*u_de3 + (1-alpha).*v_de3;
eh_de3 = sum(sum(prob.*h_de3));
% equally weighted utility
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

%% Honest equilibrium with compensation (satisfies IC constraints)
% first, check if first best solutions satisfy IC constraints
% agent 1
ic_1 = sum(sum(a1.*log(c_fb(:,2:3)./(c_fb(:,1:2)+delta1.*rho)) + (1-a1).*log(Q_fb(:,2:3)./Q_fb(:,1:2))<0));
% agent 2
ic_2 = sum(sum(a2.*log(d_fb(2:3,:)./(c_fb(1:2,:)+delta2)) + (1-a2).*log(Q_fb(2:3,:)./Q_fb(1:2,:))<0));

if ic_1 + ic_2 == 0
    c_hc = c_fb;
    d_hc = d_fb;
    Q_hc = Q_fb;
    u_hc = u_fb;
    v_hc = v_fb;
else
    honest = @(x) honest_bilateral_public_good2(x,a1,a2,price,delta1,delta2,alpha,y,rho,prob);
    rng default % For reproducibility
    ms = MultiStart('Display','iter');
    pts = linspace(0.8,1,5)'.*[c_fb(1,1) d_fb(1,1) d_fb(1,1) d_fb(1,1) c_fb(1,1) c_fb(1,1)];
    tpoints = CustomStartPointSet(pts);
    problem = createOptimProblem('fmincon','x0',[c_fb(1,1);d_fb(1,1);d_fb(1,1);d_fb(1,1);c_fb(1,1);c_fb(1,1)],...
        'objective',honest,'lb',[0;0;0;0;0;0],'ub',[y(1,1);y(1,1);y(1,2);y(1,3);y(2,1);y(3,1)],'Aineq',[1 1 0 0 0 0],'bineq',y(1,1));
    [c,~,~,~,~] = run(ms,problem,tpoints);
    [~, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_bilateral_public_good2(c,a1,a2,price,delta1,delta2,alpha,y,rho,prob);
end
eQ_hc = sum(sum(prob.*Q_hc));
% expected utility
eu_hc = sum(sum(prob.*u_hc));
ev_hc = sum(sum(prob.*v_hc));
% household utility
h_hc = alpha.*u_hc + (1-alpha).*v_hc;
eh_hc = sum(sum(prob.*h_hc));
% equally-weighted utility
sp_hc = 0.5.*u_hc + 0.5.*v_hc;
esp_hc = sum(sum(prob.*sp_hc));
% marriage surplus
ms_hc = u_hc + v_hc - h_nm;
ems_hc = sum(sum(prob.*ms_hc));
% marital surplus share
eu_ms_hc = sum(sum(prob.*(u_hc - u_nm)));
ev_ms_hc = sum(sum(prob.*(v_hc - v_nm)));
u_share_hc = eu_ms_hc./ems_hc;
v_share_hc = ev_ms_hc./ems_hc;

%% Results
% all utilities in one matrix
c = [c_fb c_hc c_de1 c_de2 c_de3];
d = [d_fb d_hc d_de1 d_de2 d_de3];
Q = [eQ_fb eQ_hc eQ_de1 eQ_de2 eQ_de3];
eu = [eu_fb eu_hc eu_de1 eu_de2 eu_de3 eu_nm];
ev = [ev_fb ev_hc ev_de1 ev_de2 ev_de3 ev_nm];
eh = [eh_fb eh_hc eh_de1 eh_de2 eh_de3 eh_nm];
esp = [esp_fb esp_hc esp_de1 esp_de2 esp_de3 esp_nm];
ems = [ems_fb ems_hc ems_de1 ems_de2 ems_de3];
u_share = [u_share_fb u_share_hc u_share_de1 u_share_de2 u_share_de3];
v_share = [v_share_fb v_share_hc v_share_de1 v_share_de2 v_share_de3];

end

% History of trying to solve the honest equilibrium
% 1. use matlab to solve the entire optimization problem: very fast but inaccurate
    %honest = @(x) -sum(sum(prob.*(alpha.*a1.*log(x(:,:,1))+(1-alpha).*a2.*x(:,:,2)+(alpha.*(1-a1)+(1-alpha).*(1-a2)).*log((y-x(:,:,1)-x(:,:,2))./price))));
    %options = optimoptions('fmincon','Display','iter');
    %nonlcon = @(x) supernonlcon(x,y,price,a1,a2,delta1,delta2,rho);
    %c = fmincon(honest,cat(3,c_fb,d_fb),[],[],[],[],zeros(3,3,2),repmat(y,1,1,2),nonlcon,options);
    %c_hc = c(:,:,1);
    %d_hc = c(:,:,2);
    %Q_hc = (y-c_hc-d_hc)./price;
    %u_hc = a1.*log(c_hc) + (1-a1).*log(Q_hc);
    %v_hc = a2.*log(d_hc) + (1-a2).*log(Q_hc);

% 2. use one function that has 6 degrees of freedom
    %honest = @(x) honest_bilateral_public_good2(x,a1,a2,price,delta1,delta2,alpha,y,rho,prob);
    %c = fminsearch(honest,[0.8*c_fb(1,1);0.8*d_fb(1,1);0.8*d_fb(1,1);0.8*d_fb(1,1);0.8*c_fb(1,1);0.8*c_fb(1,1)],options);
    %k = 0.8;
    %c3 = fmincon(honest,[k*c_fb(1,1);k*d_fb(1,1);k*d_fb(1,1);k*d_fb(1,1);k*c_fb(1,1);k*c_fb(1,1)],[],[],[],[],[0;0;0;0;0;0],[y(1,1);y(1,1);y(1,2);y(1,3);y(2,1);y(3,1)],[],options); 

% 3. same function as above, with multistart fmincon, fmincon without
% multistart returns errors
%rng default % For reproducibility
%ms = MultiStart('UseParallel',true,'Display','iter');
%problem = createOptimProblem('fmincon','x0',[c_fb(1,1);d_fb(1,1);d_fb(1,1);d_fb(1,1);c_fb(1,1);c_fb(1,1)],...
%        'objective',honest,'lb',[0;0;0;0;0;0],'ub',[y(1,1);y(1,1);y(1,2);y(1,3);y(2,1);y(3,1)],'Aineq',[1 1 0 0 0 0],'bineq',y(1,1));
%[result,~,~,~,~] = run(ms,problem,10);
%[~, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_bilateral_public_good2(result,a1,a2,price,delta1,delta2,alpha,y,rho,prob);

% 4. Use iterative function with multistart fmincon, fmincon without
% multistart returns errors
%rng default % For reproducibility
%ms = MultiStart('UseParallel',true,'Display','iter');
%problem = createOptimProblem('fmincon','x0',[c_fb(1,1);d_fb(1,1);d_fb(1,1);d_fb(1,1);c_fb(1,1);c_fb(1,1)],...
%'objective',honest,'lb',[0;0;0;0;0;0],'ub',[y(1,1);y(1,1);y(1,2);y(1,3);y(2,1);y(3,1)],'Aineq',[1 1 0 0 0 0],'bineq',y(1,1));
%[c,~,~,~,~] = run(ms,problem,10);
%c = fmincon(honest,[c_fb(1,1);d_fb(1,1)],[],[],[1 1],y(1,1),[0;0],[y(1,1);y(1,1)],[],options);

% 5. Use iterative function with fminsearch
%    honest = @(x) honest_bilateral_public_good(x,a1,a2,price,delta1,delta2,alpha,y,rho,prob);
%    options = optimset('Display','iter','TolFun',1e-3,'TolX',1e-3);
%    c = fminsearch(honest,[c_fb(1,1);d_fb(1,1)],options);
%    [~, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_bilateral_public_good(c,a1,a2,price,delta1,delta2,alpha,y,rho,prob);

    %honest = @(x) honest_bilateral_public_good(x,a1,a2,price,delta1,delta2,alpha,y,rho,prob);
    %options = optimset('Display','iter','TolFun',5e-3,'TolX',5e-3);
    %c = fminsearch(honest,[c_fb(1,1);d_fb(1,1)],options);
    %[~, c_hc, d_hc, Q_hc, u_hc, v_hc] = honest_bilateral_public_good(c,a1,a2,price,delta1,delta2,alpha,y,rho,prob);


