% Consumption savings with temporary shock (binary)

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

% Temporary income shock S
S = [0 1]';
p = 0.1;

%% Value function iteration
%Set up arrays to store value functions, consumption rules, and policy
%functions
V = zeros([length(bgrid),T+1,2]);
C = zeros([length(bgrid),T+1,2]);
G = zeros([length(bgrid),T+1,2]);
%Terminal value function
V(:,T+1,:) = 0;
%Compute V and C for each time period
for t = T:-1:1
    %Loop over all possible values of a
    for i = 1:length(bgrid)
        for k = 1:2
            % consumption
            c_test = y(t).*S(k) + bgrid(i) - bgrid'./(1+r);
            penalty = (c_test<=0);
            % Bellman equation
            EV = p.*V(:,t+1,1) + (1-p).*V(:,t+1,2);
            X = utility(c_test) + beta.*EV + penalty*(-1e50);
            %Value function is the max over X, policy function is the index
            [V(i,t,k), G(i,t,k)] = max(X);
            %Consumption is a function of a
            C(i,t,k) = y(t).*S(k) + bgrid(i) - bgrid(G(i,t,k))./(1+r);
        end
    end
end

%% Find the lifetime wealth path
% Simulate Markov Chain for S (transitory income shocks)
n = 81; s0 = 1; m = 3;
sim=rand(n,1); state_v = zeros(n,1); state_v(1)= s0;
ptran = [p 1-p
        p 1-p];
cumsum_tran=[zeros(m-1,1) cumsum(ptran')'];
for t=2:n
    state_v(t)=find(((sim(t)<=cumsum_tran(state_v(t-1),2:m))&(sim(t)>cumsum_tran(state_v(t-1),1:m-1))),1);
end
S_chain = S(state_v);
realized_inc = y'.*S_chain;

% I is a path of indices indicating optimal savings behavior.
I = zeros([1 T+1]);
I(1) = 1;
% b_path is lifetime savings path
b_path = zeros([1 T+1]);
% c_path is lifetime consumption path
c_path = zeros([1 T+1]);

for t = 1:T
    I(t+1) = G(I(t),t,state_v(t));
    b_path(t) = bgrid(I(t));
    c_path(t) = C(I(t),t,state_v(t));
end    

% plot
figure;
x = 20:100;
plot(x,realized_inc,x,c_path,x,b_path);
legend('Income','Consumption','Savings')

%% Calculate MPC
% calculate cash on hand
w40 = bgrid + y(20);
% MPC at 40
MPC40 = (C(:,20,2)-C(:,20,1))./y(20);
% calculate cash on hand
w50 = bgrid + y(30);
% MPC at 40
MPC50 = (C(:,30,2)-C(:,30,1))./y(30);
% plot MPC against cash on hand
figure;
plot(w50,MPC50,w40,MPC40)