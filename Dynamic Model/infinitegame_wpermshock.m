%% Infinite game representation of income hiding %%
% Two agents. agent 2 brings in fixed income every period. agent 1's income
% is stochastic. Agent 1 can decide to report income that's weakly lower
% than realized income. The probability of being found out is an increasing
% functio of the percentage of income agent 1 is hiding. If agent 1 hides
% and is found out, they divorce and each agent get autarky value forever.
% Otherwise, the game continues infinitely.
%
% This produces a intuitive income hiding equilibrium where hiding
% increases with 1) lower probability of detection 2) higher preference of
% private good 3) higher income compared to partner 4) lower Pareto weight
% 5) less patient 6) lower cost of hiding
%
% Dec 11, 2020
% Sally Zhang

clear all; close all; clc;

%% Parameters
% set seed
rng(4);

% Parameters
% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;

% Pareto weight
alpha = 0.5;

% Time discounting
beta = 0.96;

% Cost of hiding
delta = 0.8;

%% Functional forms
% Income process is uniform on the interval [ymin,ymax]. agent 2's income
% is fixed.
%ymin = 1; ymax = 5;
y2 = 3;

% AR(1) income process, discretized using Tauchen method
% Record parameters. epsilon_t is normal(mu,sigma^2) where theta^2 =
% sigma^2/(1-phi^2).
mu = 3; sigma = 0.15; phi = 1.02; theta = sigma.^2./(1-phi.^2);
% Pick (m-1) gridpoints such that the successive areas under the standard
%nomrmal are equal to 1/m.
m = 9; a = zeros([m+1 1]);
a(1) = -Inf; a(m+1) = Inf;
for i = 2:m
    a(i) = norminv((i-1)/m);
end
% Calculate the transition matrix following Deaton(1991)
Trans = zeros(m-1);
for i = 1:m-1
    for j = 1:m-1
        f = @(x) exp(-x.^2./(2.*theta.^2)).*(normcdf((theta.*a(j+1)-phi.*x)./sigma)-normcdf((theta.*a(j)-phi.*x)./sigma));
        int = integral(f,theta*a(i),theta*a(i+1));
        Trans(i,j) = (1/(theta*sqrt(2*pi)))*int;
    end
end
% Scale all the rows such that they each sum to one
sumT = sum(Trans,2);
Trans = Trans./sumT;
% Income shock in each state
epsilon = theta.*a(2:m) + mu;

% probability of punishment (square)
prob = @(p) p.^2;
% probability of punishment (logit)
%prob = @(p) (p~=0).*0.2.*exp(p)./(1+exp(p));
%prob_prime = @(p) exp(p)./((1+exp(p)).^2);
% graph probability of punishment
%figure;
%test = linspace(0,ymax,100);
%plot(test, prob(test))

% Utility depends on actual income and realized income
gamma = (1-a1).*alpha + (1-a2).*(1-alpha);
u = @(income,z) a1.*log(alpha.*a1.*(y2+z) + delta.*(income-z)) + (1-a1).*log(gamma.*(y2+z));

% Autarky value
% Simulate Markov Chain for epsilon (permanent income shocks)
n = 2000; s0 = 4;
sim=rand(n,1); state_e = zeros(n,1); state_e(1)= s0;
cumsum_tran=[zeros(m-1,1) cumsum(Trans')'];
for t=2:n
    state_e(t)=find(((sim(t)<=cumsum_tran(state_e(t-1),2:m))&(sim(t)>cumsum_tran(state_e(t-1),1:m-1))),1);
end
epsilon_chain = epsilon(state_e);
% calculate utility for each realization
simvaut1 = a1.*log(a1.*epsilon_chain)+(1-a1).*log((1-a1).*epsilon_chain);
% calculate expectation
vaut1 = mean(simvaut1)./(1-beta);

%% Calculate best response
% number of discretized states
n = 500;
% set up matrices to store results
income = linspace(1,5,n);
maxv = zeros(1,n);
maxz = zeros(1,n);
% guess of expected continuation value
V_exp = vaut1+1;

% initialize difference
diff = 1;
% error tolerance
error = 1e-4;

% use a loop to converge on iteration
while diff>error
    % for each state of income realization
    for i = 1:m-1
        % this period's utility
        u_test = @(z) u(epsilon(i),z);
        % continuation value
        V = @(z) u_test(z) + beta.*vaut1.*prob((epsilon(i)-z)./epsilon(i)) + beta.*(1-prob((epsilon(i)-z)./epsilon(i)).*V_exp;
        % discretize possible reported income space
        z_test = linspace(0,income(i),500);
        % choose reported income to maxize continuation value
        V_test = V(z_test);
        % record choices
        [maxv(i), maxzind] = max(V_test);
        maxz(i) = z_test(maxzind);
    end
    % Calculate expectation
    next_value1 = zeros(length(d_hat),m-1);
    next_value2 = zeros(length(d_hat),m-1);
    for l = 1:m-1
        next_value1(:,l) = power_epsilon(l).*V_initial(:,l,1).*g^(1-rho);
        next_value2(:,l) = power_epsilon(l).*V_initial(:,l,2).*g^(1-rho);
    end
    expectation = next_value1*Trans(:,1);
    % update expected continuation value
    diff = V_exp - mean(maxv);
    V_exp = mean(maxv);
end

%% Policy functions
c1 = a1.*alpha.*(maxz+y2) + delta.*(income-maxz);
c2 = a2.*(1-alpha).*(maxz+y2);
Q = gamma.*(maxz+y2);

c1_fb = a1.*alpha.*(income+y2);
c2_fb = a2.*(1-alpha).*(income+y2);
Q_fb = gamma.*(income+y2);

figure;
subplot(2,2,1)
plot(income,maxz,income,income);
xlabel('Income');
ylabel('Reported Income');
legend('hiding','no hiding');
title('Reported Income')

subplot(2,2,2)
plot(income,c1,income,c1_fb);
xlabel('Income');
ylabel('Consumption');
legend('hiding','no hiding');
title('C1');

subplot(2,2,3)
plot(income,c2,income,c2_fb);
xlabel('Income');
ylabel('Consumption');
legend('hiding','no hiding');
title('C2');

subplot(2,2,4)
plot(income,Q,income,Q_fb);
xlabel('Income');
ylabel('Consumption');
legend('hiding','no hiding');
title('Q');

%% MPC
% marginal propensity to report
MPP = mean((maxz(2:end) - maxz(1:end-1))./(income(2:end) - income(1:end-1)));
% marginal propensity to consume on c1
MPC1 = mean((c1(2:end) - c1(1:end-1))./(income(2:end) - income(1:end-1)));
MPC1_fb = mean((c1_fb(2:end) - c1_fb(1:end-1))./(income(2:end) - income(1:end-1)));
% marginal propensity to consume on c2
MPC2 = mean((c2(2:end) - c2(1:end-1))./(income(2:end) - income(1:end-1)));
MPC2_fb = mean((c2_fb(2:end) - c2_fb(1:end-1))./(income(2:end) - income(1:end-1)));
% marginal propensity to consume on Q
MPCQ = mean((Q(2:end) - Q(1:end-1))./(income(2:end) - income(1:end-1)));
MPCQ_fb = mean((Q_fb(2:end) - Q_fb(1:end-1))./(income(2:end) - income(1:end-1)));

%% Simulation
% simulation time periods
T = 100;

% simulate income stream
ind_path = randi([1 n],T,1);
y_path = income(ind_path);

% autarky utility stream
aut1_path = a1.*log(a1.*y_path)+(1-a1).*log((1-a1).*y_path);
aut2 = a2.*log(a2.*y2)+(1-a2).*log((1-a2).*y2);

% simulate reported income stream
z_path = maxz(ind_path);

% simulate utility stream
c1_path = alpha.*a1.*(y2+z_path) + delta.*(y_path-z_path);
c2_path = (1-alpha).*a2.*(y2+z_path);
Q_path = gamma.*(y2+z_path);
u_path = a1.*log(c1_path) + (1-a1).*log(Q_path);
v_path = a2.*log(c2_path) + (1-a2).*log(Q_path);

% simulate probabilities of detection
%prob_path = prob(y_path-z_path);
prob_path = prob((y_path-z_path)./y_path);
% draw random numbers to see whether punishment state realizes
rand_path = rand(1,T);
punishment_path = (rand_path<prob_path);
% first punishment state
first_punishment = find(punishment_path,1);

% all time periods after the first punishment state is autarky
c1_path(first_punishment+1:end) = a1.*y_path(first_punishment+1:end);
c2_path(first_punishment+1:end) = a2.*2;
Q_path(first_punishment+1:end) = (1-a1).*y_path(first_punishment+1:end) + (1-a2).*2;
u_path(first_punishment+1:end) = aut1_path(first_punishment+1:end);
v_path(first_punishment+1:end) = aut2;

% calculate marriage surplus streams
surplus_path = u_path + v_path - aut1_path - aut2;

%% Simulation plots
figure;
subplot(2,2,1);
plot(1:T,y_path,1:T,z_path);
legend('Income','Reported Income');
xlabel('Time');
ylabel('Income');
title('Income');

subplot(2,2,2);
plot(1:T,y_path-z_path);
xlabel('Time');
ylabel('Income');
title('Hiding');

subplot(2,2,3);
plot(1:T,c1_path,1:T,c2_path,1:T,Q_path);
xlabel('Time');
ylabel('Consumption');
legend('C1','C2','Q');
title('Consumption');

subplot(2,2,4);
plot(1:T,surplus_path);
xlabel('Time');
ylabel('Utility');
title('Marital Surplus');

%% Repeated Simulation
