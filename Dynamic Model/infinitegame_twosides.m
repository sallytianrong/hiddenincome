%{
function [mean_MPP1, mean_MPP2, mean_MPC1_inc1, mean_MPC1_inc2, mean_MPC2_inc1, mean_MPC2_inc2, mean_MPCQ_inc1, mean_MPCQ_inc2,...
    mean_MPC1_inc1_fb, mean_MPC1_inc2_fb, mean_MPC2_inc1_fb, mean_MPC2_inc2_fb, mean_MPCQ_inc1_fb, mean_MPCQ_inc2_fb] =...
    infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,beta1,beta2,delta1,delta2)
%}

function [z1max, z2max] = infinitegame_twosides(a1,a2,y1min,y1max,y2min,y2max,alpha,beta,delta1,delta2,price,n)

%% Infinite game representation of income hiding %%
% Two agents, both agents' income are stochastic. is stochastic. Both agents
% can report income that's weakly lower than realized income. The probability
% of being found out is an increasing function of the percentage of hidden
% income. If one hides income and is found out, they divorce and each agent
% get autarky value forever. Otherwise, the game continues infinitely.
%
% This produces a intuitive income hiding equilibrium where hiding
% increases with 1) lower probability of detection 2) higher preference of
% private good 3) higher income compared to partner 4) lower Pareto weight
% 5) less patient 6) lower cost of hiding
%
% Dec 15, 2020
% Sally Zhang

%{
clear all; close all; clc;

%% Parameters
% set seed
rng(4);

% Preference of private good versus public good
a1 = 0.5; a2 = 0.5;

% Two income processes: uniform
y1min = 1; y1max = 5;
y2min = 1; y2max = 5;

% Pareto weight
alpha = 0.5;

% Time discounting
beta1 = 0.9; beta2 = 0.9;

% Cost of hiding
delta1 = 0.8; delta2 = 0.8;
%}

%% functional forms
% probability of punishment (logit)
%prob = @(p) (p~=0).*0.1.*exp(p)./(1+exp(p));
%prob_prime = @(p) exp(p)./((1+exp(p)).^2);
% plot probability
%test = linspace(0,y1max,100);
%plot(test, prob(test))

% utility
gamma = (1-a1).*alpha + (1-a2).*(1-alpha);
% each agent's utility depend on their actual income and both agents'
% reported income
u1 = @(y,z1,z2) a1.*log(alpha.*a1.*(z1+z2) + delta1.*(y-z1)) + (1-a1).*log(gamma.*(z1+z2)./price);
u2 = @(y,z1,z2) a2.*log((1-alpha).*a2.*(z1+z2) + delta2.*(y-z2)) + (1-a2).*log(gamma.*(z1+z2)./price);

% discretize income space
%n = 10;
income1 = linspace(y1min,y1max,n);
income2 = linspace(y2min,y2max,n);

% Autarky value
% autarky utility
uaut1 = @(y) a1.*log(a1.*y)+(1-a1).*log((1-a1).*y./price);
uaut2 = @(y) a2.*log(a2.*y)+(1-a2).*log((1-a2).*y./price);
% calculate expected utility under forever autarky
vaut1 = mean(uaut1(income1))./(1-beta);
vaut2 = mean(uaut2(income2))./(1-beta);

% error tolerance
error = 1e-4;

%% Agent 1
% set up matrices for agent 1
maxv1 = zeros(n);
maxzind1 = zeros(n);
% guess of initial continuation value
diff = 1;
V_exp1 = vaut1+1;

% Agent 1
tic
while diff>error
    % for each self income realization
    for i = 1:n
        % and each reported income of the other agent
        for j = 1:n
        % possible reported income is not greater than actual income
        z_test = income1(1:i)';
        % current period utility
        u1_test = u1(income1(i),z_test,income2(j));
        % continuation value
        V_test = u1_test + beta.*vaut1.*((income1(i)-z_test)./income1(i)).^2 + beta.*(1-((income1(i)-z_test)./income1(i)).^2).*V_exp1;
        % choose reported income to maxize continuation value
        [maxv1(i,j), maxzind1(i,j)] = max(V_test);
        end
    end
    diff = abs(V_exp1 - mean(mean(maxv1)));
    V_exp1 = mean(mean(maxv1));
end
toc

%% Agent 2
% set up matrices for agent 2
maxv2 = zeros(n);
maxzind2 = zeros(n);

% initial guess of continuation value
diff = 1;
V_exp2 = vaut2+1;

tic
% Agent 2
while diff>error
    % for each self income realization
    for i = 1:n
    %for i = 1:n
        % and each reported income of the other agent
        for j = 1:n
        % possible reported income is not greater than actual income
        z_test = income2(1:i)';
        % current period utility
        u2_test = u2(income2(i),income1(j),z_test);
        % continuation value
        V_test = u2_test + beta.*vaut2.*((income2(i)-z_test)./income2(i)).^2 + beta.*(1-((income2(i)-z_test)./income2(i)).^2).*V_exp2;
        % choose reported income to maximize continuation value
        [maxv2(i,j), maxzind2(i,j)] = max(V_test);
        end
    end
    diff = abs(V_exp2 - mean(mean(maxv2)));
    V_exp2 = mean(mean(maxv2));
end
toc

%% Find best response
z1 = zeros(n);
z2 = zeros(n);
tic
% for each agent 1's income realization
for i = 1:n
    % and each agent 2's income realization
    for j = 1:n
        % guess a response function
        z1(i,j) = i;
        z2(i,j) = j;
        max_diff = 100;
        while max_diff>1
            % calculate difference between current guess and best response
            z1_diff = abs(z1(i,j)-maxzind1(i,z2(i,j)));
            z2_diff = abs(z2(i,j)-maxzind2(j,z1(i,j)));
            max_diff = max(z1_diff,z2_diff);
            % update
            z1(i,j) = maxzind1(i,z2(i,j));
            z2(i,j) = maxzind2(j,z1(i,j));
        end
    end
end
toc

%% Policy functions
% Reported income as a function of realized income
z1max = income1(z1);
z2max = income2(z2);

% consumption
income1_rep = repmat(income1',1,n);
income2_rep = repmat(income2,n,1);
c1 = a1.*alpha.*(z1max + z2max) + (income1_rep - z1max);
c2 = a2.*(1-alpha).*(z1max + z2max) + (income2_rep - z2max);
Q = gamma.*(z1max + z2max);

% utility
u1_pol = u1(income1_rep,z1max,z2max);
u2_pol = u2(income2_rep,z1max,z2max);

% no hiding
c1_fb = a1.*alpha.*(income1_rep + income2_rep);
c2_fb = a2.*(1-alpha).*(income1_rep + income2_rep);
Q_fb = gamma.*(income1_rep + income2_rep);

%{
%plot
figure;
subplot(2,2,1);
plot(income1,income1,income1,z1max(:,20),income1,z1max(:,100));
legend('No Hiding','Agent 2 low inc', 'Agent 2 high inc');
xlabel('Income of Agent 1');

subplot(2,2,2);
plot(income1, c1_fb(:,20), income1, c1_fb(:,100), income1,c1(:,20),income1,c1(:,100));
legend('No Hiding Low inc', 'No Hiding High Inc', 'Agent 2 low inc', 'Agent 2 high inc');
xlabel('Income of Agent 1');

subplot(2,2,3);
plot(income1, c2_fb(:,20), income1, c2_fb(:,100), income1,c2(:,20),income1,c2(:,100));
legend('No Hiding Low inc', 'No Hiding High Inc', 'Agent 2 low inc', 'Agent 2 high inc');
xlabel('Income of Agent 1');

subplot(2,2,4);
plot(income1, Q_fb(:,20), income1, Q_fb(:,100), income1,Q(:,20),income1,Q(:,100));
legend('No Hiding Low inc', 'No Hiding High Inc', 'Agent 2 low inc', 'Agent 2 high inc');
xlabel('Income of Agent 1');

figure;
subplot(2,2,1);
plot(income2,income2,income2,z2max(20,:),income2,z2max(100,:));
legend('No Hiding','Agent 1 low inc', 'Agent 1 high inc');
xlabel('Income of Agent 2');

subplot(2,2,2);
plot(income2, c1_fb(20,:), income2, c1_fb(100,:), income2,c1(20,:),income2,c1(100,:));
legend('No Hiding Low inc', 'No Hiding High Inc', 'Agent 1 low inc', 'Agent 1 high inc');
xlabel('Income of Agent 2');

subplot(2,2,3);
plot(income2, c2_fb(20,:), income2, c2_fb(100,:), income2,c2(20,:),income2,c2(100,:));
legend('No Hiding Low inc', 'No Hiding High Inc', 'Agent 1 low inc', 'Agent 1 high inc');
xlabel('Income of Agent 2');

subplot(2,2,4);
plot(income2, Q_fb(20,:), income2, Q_fb(100,:), income2,Q(20,:),income2,Q(100,:));
legend('No Hiding Low inc', 'No Hiding High Inc', 'Agent 1 low inc', 'Agent 1 high inc');
xlabel('Income of Agent 2');
%}

%{
%% Marginal Propensities
% marginal propensity to report
MPP1 = zeros(1,n);
MPP2 = zeros(1,n);
% marginal propensity to consume as a function of agent 1 and agent 2's
% income
MPC1_inc1 = zeros(1,n);
MPC1_inc2 = zeros(1,n);
MPC2_inc1 = zeros(1,n);
MPC2_inc2 = zeros(1,n);
MPCQ_inc1 = zeros(1,n);
MPCQ_inc2 = zeros(1,n);
% marginal propensity to consume as a function of agent 1 and agent 2's
% income
MPC1_inc1_fb = zeros(1,n);
MPC1_inc2_fb = zeros(1,n);
MPC2_inc1_fb = zeros(1,n);
MPC2_inc2_fb = zeros(1,n);
MPCQ_inc1_fb = zeros(1,n);
MPCQ_inc2_fb = zeros(1,n);
for i = 1:n
    MPP1(i) = mean((z1max(2:end,i) - z1max(1:end-1,i))./(income1(2:end) - income1(1:end-1))');
    MPP2(i) = mean((z2max(i,2:end) - z2max(i,1:end-1))./(income2(2:end) - income2(1:end-1)));
    MPC1_inc1(i) = mean((c1(2:end,i) - c1(1:end-1,i))./(income1(2:end) - income1(1:end-1))');
    MPC1_inc2(i) = mean((c1(i,2:end) - c1(i,1:end-1))./(income2(2:end) - income2(1:end-1)));
    MPC2_inc1(i) = mean((c2(2:end,i) - c2(1:end-1,i))./(income1(2:end) - income1(1:end-1))');
    MPC2_inc2(i) = mean((c2(i,2:end) - c2(i,1:end-1))./(income2(2:end) - income2(1:end-1)));
    MPCQ_inc1(i) = mean((Q(2:end,i) - Q(1:end-1,i))./(income1(2:end) - income1(1:end-1))');
    MPCQ_inc2(i) = mean((Q(i,2:end) - Q(i,1:end-1))./(income2(2:end) - income2(1:end-1)));
    MPC1_inc1_fb(i) = mean((c1_fb(2:end,i) - c1_fb(1:end-1,i))./(income1(2:end) - income1(1:end-1))');
    MPC1_inc2_fb(i) = mean((c1_fb(i,2:end) - c1_fb(i,1:end-1))./(income2(2:end) - income2(1:end-1)));
    MPC2_inc1_fb(i) = mean((c2_fb(2:end,i) - c2_fb(1:end-1,i))./(income1(2:end) - income1(1:end-1))');
    MPC2_inc2_fb(i) = mean((c2_fb(i,2:end) - c2_fb(i,1:end-1))./(income2(2:end) - income2(1:end-1)));
    MPCQ_inc1_fb(i) = mean((Q_fb(2:end,i) - Q_fb(1:end-1,i))./(income1(2:end) - income1(1:end-1))');
    MPCQ_inc2_fb(i) = mean((Q_fb(i,2:end) - Q_fb(i,1:end-1))./(income2(2:end) - income2(1:end-1)));

end

% mean marginal propensities
mean_MPP1 = mean(MPP1);
mean_MPP2 = mean(MPP2);
mean_MPC1_inc1 = mean(MPC1_inc1);
mean_MPC1_inc2 = mean(MPC1_inc2);
mean_MPC2_inc1 = mean(MPC2_inc1);
mean_MPC2_inc2 = mean(MPC2_inc2);
mean_MPCQ_inc1 = mean(MPCQ_inc1);
mean_MPCQ_inc2 = mean(MPCQ_inc2);
mean_MPC1_inc1_fb = mean(MPC1_inc1_fb);
mean_MPC1_inc2_fb = mean(MPC1_inc2_fb);
mean_MPC2_inc1_fb = mean(MPC2_inc1_fb);
mean_MPC2_inc2_fb = mean(MPC2_inc2_fb);
mean_MPCQ_inc1_fb = mean(MPCQ_inc1_fb);
mean_MPCQ_inc2_fb = mean(MPCQ_inc2_fb);
%}
