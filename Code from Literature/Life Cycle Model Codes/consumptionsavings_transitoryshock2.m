% Consumption savings with temporary shock (nonbinary)

%% Set up
%Declare parameters
beta = 0.98; r = 0.02; sigma = 0.5; T = 80; rho = 2;

%Define y_t. Note that Y(t) as defined in the matrix is actually y_(t-1).
%This indexing change holds for all of y_t, b_t, and a_t.
y = zeros([1 T+1]);
for t = 1:46
    y(t) = -3 + 0.25*(t-1+20) - 0.0025*(t-1+20)^2;
end

%Define b_t.
B = zeros([1 T+1]);
B(1) = 0;

%utility function
utility = @(c) c.^(1-rho)./(1-rho);

%Savings grid
n = 1000; bgrid = linspace(1e-3,60,n);

%% Discretize transitory shock
% Pick (m-1) gridpoints such that the successive areas under the standard
%normal are equal to 1/m.
m = 9; epsilon = zeros(1,m);
for i = 1:m
    epsilon(i) = norminv(i/(m+1),0,sigma);
end

%% Value function iteration
%Set up arrays to store value functions, consumption rules, and policy
%functions
V = zeros([length(bgrid),T+1,m]);
C = zeros([length(bgrid),T+1,m]);
G = zeros([length(bgrid),T+1,m]);
%Terminal value function
V(:,T+1,:) = 0;
%Compute V and C for each time period
for t = T:-1:1
    %Loop over all possible values of a
    for i = 1:length(bgrid)
        % for all possible states of the world
        for k = 1:m
            % consumption
            c_test = y(t).*exp(epsilon(k)) + bgrid(i) - bgrid'./(1+r);
            penalty = (c_test<=0);
            % Bellman equation
            EV = zeros(length(bgrid),1);
            for l = 1:m
                EV = EV + (1./m).*V(:,t+1,l);
            end
            X = utility(c_test) + beta.*EV + penalty*(-1e50);
            %Value function is the max over X, policy function is the index
            [V(i,t,k), G(i,t,k)] = max(X);
            %Consumption is a function of a
            C(i,t,k) = y(t).*exp(epsilon(k)) + bgrid(i) - bgrid(G(i,t,k))./(1+r);
        end
    end
end

%% Find the lifetime wealth path
% Simulate Markov Chain for epsilon (transitory income shocks)
sim=randi(m,[1,T+1]); % random numbers
epsilon_path = epsilon(sim);
realized_inc = y.*exp(epsilon_path);

% I is a path of indices indicating optimal savings behavior.
I = zeros([1 T+1]);
I(1) = 1;
% b_path is lifetime savings path
b_path = zeros([1 T+1]);
% c_path is lifetime consumption path
c_path = zeros([1 T+1]);

for t = 1:T
    I(t+1) = G(I(t),t,sim(t));
    b_path(t) = bgrid(I(t));
    c_path(t) = C(I(t),t,sim(t));
end    

% plot
figure;
x = 20:100;
plot(x,y,x,realized_inc,x,c_path,x,b_path);
legend('Permanent Income','Income','Consumption','Savings')

%% Calculate MPC
% calculate cash on hand
w40 = bgrid + realized_inc(20);
% MPC at 40
state40 = sim(20);
MPC40pos = (C(:,20,state40+1)-C(:,20,state40))./w40';
MPC40neg = (C(:,20,state40)-C(:,20,state40-1))./w40';
% plot MPC against cash on hand
figure;
plot(w40,MPC40pos,w40,MPC40neg)